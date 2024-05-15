import argparse
from pkg_resources import resource_filename
from pangolin.model import *
import vcf
import gffutils
import pandas as pd
import pyfastx
# import time
# startTime = time.time()

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


def compute_score(ref_seq, alt_seq, strand, d, models):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

    if torch.cuda.is_available():
        ref_seq = ref_seq.to(torch.device("cuda"))
        alt_seq = alt_seq.to(torch.device("cuda"))

    pangolin = []
    for j in range(4):
        score = []
        for model in models[3*j:3*j+3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                alt = model(alt_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                if strand == '-':
                    ref = ref[::-1]
                    alt = alt[::-1]
                l = 2*d+1
                ndiff = np.abs(len(ref)-len(alt))
                if len(ref)>len(alt):
                    alt = np.concatenate([alt[0:l//2+1],np.zeros(ndiff),alt[l//2+1:]])
                elif len(ref)<len(alt):
                    alt = np.concatenate([alt[0:l//2],np.max(alt[l//2:l//2+ndiff+1], keepdims=True),alt[l//2+ndiff+1:]])
                score.append(alt-ref)
        pangolin.append(np.mean(score, axis=0))
    
    pangolin = np.array(pangolin)
    loss = pangolin[np.argmin(pangolin, axis=0), np.arange(pangolin.shape[1])]
    gain = pangolin[np.argmax(pangolin, axis=0), np.arange(pangolin.shape[1])]
    return loss, gain

def compute_scores_on_batch(ref_seqs, alt_seqs, strands, d, models):

    if len(ref_seqs) != len(alt_seqs) or len(alt_seqs) != len(strands) or len(strands) != len(ref_seqs):
        raise ValueError(f'length of ref_seqs={len(ref_seqs)}, length of alt_seqs={len(alt_seqs)}, length of strands = {len(strands)}, but must be all equal')
    batch_size = len(ref_seqs)
    
    encoded_refs = []
    encoded_alts = []
    
    for i in range(len(ref_seqs)):
        strand = strands[i]
        ref_seq = torch.from_numpy(one_hot_encode(ref_seqs[i], strand).T).float()
        alt_seq = torch.from_numpy(one_hot_encode(alt_seqs[i], strand).T).float()
        encoded_refs.append(ref_seq)
        encoded_alts.append(alt_seq)
        
    batch_ref = torch.stack(encoded_refs)
    batch_alt = torch.stack(encoded_alts)
    
    if torch.cuda.is_available():
        batch_ref = batch_ref.to(torch.device("cuda"))
        batch_alt = batch_alt.to(torch.device("cuda"))

    pangolin = []
    for j in range(4):
        score = []
        for model in models[3*j:3*j+3]:
            with torch.no_grad():
                pred_ref = model(batch_ref)[:,[1,4,7,10][j],:].cpu().numpy() # [0][[1,4,7,10][j],:].cpu().numpy()
                pred_alt = model(batch_alt)[:,[1,4,7,10][j],:].cpu().numpy() # [0][[1,4,7,10][j],:].cpu().numpy()
                batch_score = []
                for k in range(batch_size):
                    ref = pred_ref[k]
                    alt = pred_alt[k]
                    if strands[k] == '-':
                        ref = ref[::-1]
                        alt = alt[::-1]
                    l = 2*d+1
                    ndiff = np.abs(len(ref)-len(alt))
                    if len(ref)>len(alt):
                        alt = np.concatenate([alt[0:l//2+1],np.zeros(ndiff),alt[l//2+1:]])
                    elif len(ref)<len(alt):
                        alt = np.concatenate([alt[0:l//2],np.max(alt[l//2:l//2+ndiff+1], keepdims=True),alt[l//2+ndiff+1:]])
                    batch_score.append(alt-ref)
                score.append(batch_score)
        pangolin.append(score)
    
    pangolin = np.array(pangolin)
    pangolin = np.mean(pangolin, axis=1)
    loss_array = []
    gain_array = []
    for i in range(batch_size):
        sample = pangolin[:,i,:]
        loss = sample[np.argmin(sample, axis=0), np.arange(sample.shape[1])]
        gain = sample[np.argmax(sample, axis=0), np.arange(sample.shape[1])]
        loss_array.append(loss)
        gain_array.append(gain)
    return np.array(loss_array), np.array(gain_array)

def get_genes(chr, pos, gtf):
    genes = gtf.region((chr, pos-1, pos-1), featuretype="gene")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        gene_id = gene["gene_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == '+':
            genes_pos[gene_id] = exons
        elif gene[6] == '-':
            genes_neg[gene_id] = exons

    return (genes_pos, genes_neg)


def process_variant(lnum, chr, pos, ref, alt, gtf, models, args):
    d = args.distance
    cutoff = args.score_cutoff

    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant format not supported.")
        return -1
    elif len(ref) > 2*d:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Deletion too large")
        return -1

    fasta = pyfastx.Fasta(args.reference_file)
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq
    except Exception as e:
        print(e)
        print("[Line %s]" % lnum, "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
                                  "See error message above.")
        return -1    

    if seq[5000+d:5000+d+len(ref)] != ref:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
              % (seq[5000+d:5000+d+len(ref)], ref))
        return -1

    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos)+len(genes_neg)==0:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return -1

    # get splice scores
    loss_pos, gain_pos = None, None
    if len(genes_pos) > 0:
        loss_pos, gain_pos = compute_score(ref_seq, alt_seq, '+', d, models)
    loss_neg, gain_neg = None, None
    if len(genes_neg) > 0:
        loss_neg, gain_neg = compute_score(ref_seq, alt_seq, '-', d, models)

    scores_list = []
    for (genes, loss, gain) in (
        (genes_pos,loss_pos,gain_pos),(genes_neg,loss_neg,gain_neg)
    ):
        # Emit a bundle of scores/warnings per gene; join them all later
        for gene, positions in genes.items():
            per_gene_scores = []
            warnings = "Warnings:"
            positions = np.array(positions)
            positions = positions - (pos - d)

            if args.mask == "True" and len(positions) != 0:
                positions_filt = positions[(positions>=0) & (positions<len(loss))]
                # set splice gain at annotated sites to 0
                gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                # set splice loss at unannotated sites to 0
                not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                loss[not_positions] = np.maximum(loss[not_positions], 0)

            elif args.mask == "True":
                warnings += "NoAnnotatedSitesToMaskForThisGene"
                loss[:] = np.maximum(loss[:], 0)

            if args.score_exons == "True":
                scores1 = [gene + '_sites1']
                scores2 = [gene + '_sites2']
            
                for i in range(len(positions)//2):
                    p1, p2 = positions[2*i], positions[2*i+1]
                    if p1<0 or p1>=len(loss):
                        s1 = "NA"
                    else:
                        s1 = [loss[p1],gain[p1]]
                        s1 = round(s1[np.argmax(np.abs(s1))],2)
                    if p2<0 or p2>=len(loss):
                        s2 = "NA"
                    else:
                        s2 = [loss[p2],gain[p2]]
                        s2 = round(s2[np.argmax(np.abs(s2))],2)
                    if s1 == "NA" and s2 == "NA":
                        continue
                    scores1.append(f"{p1-d}:{s1}")
                    scores2.append(f"{p2-d}:{s2}")
                per_gene_scores += scores1 + scores2

            elif cutoff != None:
                per_gene_scores.append(gene)
                l, g = np.where(loss<=-cutoff)[0], np.where(gain>=cutoff)[0]
                for p, s in zip(np.concatenate([g-d,l-d]), np.concatenate([gain[g],loss[l]])):
                    per_gene_scores.append(f"{p}:{round(s,2)}")

            else:
                per_gene_scores.append(gene)
                l, g = np.argmin(loss), np.argmax(gain),
                gain_str = f"{g-d}:{round(gain[g],2)}"
                loss_str = f"{l-d}:{round(loss[l],2)}"
                per_gene_scores += [gain_str, loss_str]

            per_gene_scores.append(warnings)
            scores_list.append('|'.join(per_gene_scores))

    return ','.join(scores_list)

def process_batch_variants(positions, ref_seqs, alt_seqs, genes_pos_array, genes_neg_array, gtf, models, args):
    d = args.distance
    cutoff = args.score_cutoff

    batch_ref_seq = []
    batch_alt_seq = []
    batch_strand = []
    batch_genes_pos = []
    batch_genes_neg = []
    batch_positions = []

    for i in range(len(ref_seqs)):
        genes_pos = genes_pos_array[i]
        genes_neg = genes_neg_array[i]
        
        if len(genes_pos) > 0:
            batch_ref_seq.append(ref_seqs[i])
            batch_alt_seq.append(alt_seqs[i])
            batch_strand.append('+')
            batch_genes_pos.append(genes_pos)
            batch_genes_neg.append(genes_neg)
            batch_positions.append(positions[i])
        if len(genes_neg) > 0:
            batch_ref_seq.append(ref_seqs[i])
            batch_alt_seq.append(alt_seqs[i])
            batch_strand.append('-')
            batch_genes_pos.append(genes_pos)
            batch_genes_neg.append(genes_neg)
            batch_positions.append(positions[i])

    batch_loss, batch_gain = compute_scores_on_batch(batch_ref_seq, batch_alt_seq, batch_strand, d, models)
    assert batch_loss.shape[0] == len(batch_ref_seq)
    assert batch_gain.shape[0] == len(batch_ref_seq)

    batch_scores = []
    skip = False
    for k in range(len(batch_ref_seq)):
        if skip:
            skip = False
            continue

        strand = batch_strand[k]
        pos = batch_positions[k]
        if strand == '+':
            genes_pos = batch_genes_pos[k]
            loss_pos = batch_loss[k]
            gain_pos = batch_gain[k]
            try:
                both_strands = (batch_ref_seq[k] == batch_ref_seq[k+1])
            except IndexError:
                both_strands = False
            if both_strands:
                skip = True
                genes_neg = batch_genes_neg[k+1]
                loss_neg = batch_loss[k+1]
                gain_neg = batch_gain[k+1]
            else:
                genes_neg = {}
                loss_neg = None
                gain_neg = None
        else:
            genes_neg = batch_genes_neg[k]
            loss_neg = batch_loss[k]
            gain_neg = batch_gain[k]
            genes_pos = {}
            loss_pos = None
            gain_pos = None
            # try:
            #     assert batch_ref_seq[k] != batch_ref_seq[k+1]
            # except IndexError:
            #     pass
        
        scores_list = []
        for (genes, loss, gain) in (
            (genes_pos,loss_pos,gain_pos),(genes_neg,loss_neg,gain_neg)
        ):
            # Emit a bundle of scores/warnings per gene; join them all later
            for gene, positions in genes.items():
                per_gene_scores = []
                warnings = "Warnings:"
                positions = np.array(positions)
                positions = positions - (pos - d)

                if args.mask == "True" and len(positions) != 0:
                    positions_filt = positions[(positions>=0) & (positions<len(loss))]
                    # set splice gain at annotated sites to 0
                    gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                    # set splice loss at unannotated sites to 0
                    not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                    loss[not_positions] = np.maximum(loss[not_positions], 0)

                elif args.mask == "True":
                    warnings += "NoAnnotatedSitesToMaskForThisGene"
                    loss[:] = np.maximum(loss[:], 0)

                if args.score_exons == "True":
                    scores1 = [gene + '_sites1']
                    scores2 = [gene + '_sites2']

                    for i in range(len(positions)//2):
                        p1, p2 = positions[2*i], positions[2*i+1]
                        if p1<0 or p1>=len(loss):
                            s1 = "NA"
                        else:
                            s1 = [loss[p1],gain[p1]]
                            s1 = round(s1[np.argmax(np.abs(s1))],2)
                        if p2<0 or p2>=len(loss):
                            s2 = "NA"
                        else:
                            s2 = [loss[p2],gain[p2]]
                            s2 = round(s2[np.argmax(np.abs(s2))],2)
                        if s1 == "NA" and s2 == "NA":
                            continue
                        scores1.append(f"{p1-d}:{s1}")
                        scores2.append(f"{p2-d}:{s2}")
                    per_gene_scores += scores1 + scores2

                elif cutoff != None:
                    per_gene_scores.append(gene)
                    l, g = np.where(loss<=-cutoff)[0], np.where(gain>=cutoff)[0]
                    for p, s in zip(np.concatenate([g-d,l-d]), np.concatenate([gain[g],loss[l]])):
                        per_gene_scores.append(f"{p}:{round(s,2)}")

                else:
                    per_gene_scores.append(gene)
                    l, g = np.argmin(loss), np.argmax(gain),
                    gain_str = f"{g-d}:{round(gain[g],2)}"
                    loss_str = f"{l-d}:{round(loss[l],2)}"
                    per_gene_scores += [gain_str, loss_str]

                per_gene_scores.append(warnings)
                scores_list.append('|'.join(per_gene_scores))

        batch_scores.append(','.join(scores_list))

    return batch_scores

def prepare_variant_for_batch(lnum, chr, pos, ref, alt, gtf, fasta, d):
    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant format not supported.")
        return -1, -1, -1, -1
    
    elif len(ref) > 2*d:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Deletion too large")
        return -1, -1, -1, -1
    
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]
    
    try:
        seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq
    except Exception as e:
        print(e)
        print("[Line %s]" % lnum, "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
                                          "See error message above.")
        return -1, -1, -1, -1
    
    if seq[5000+d:5000+d+len(ref)] != ref:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
                % (seq[5000+d:5000+d+len(ref)], ref))
        return -1, -1, -1, -1
    
    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]
    
    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos)+len(genes_neg)==0:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return -1, -1, -1, -1
    
    return ref_seq, alt_seq, genes_pos, genes_neg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option).")
    parser.add_argument("reference_file", help="FASTA file containing a reference genome sequence.")
    parser.add_argument("annotation_file", help="gffutils database file. Can be generated using create_db.py.")
    parser.add_argument("output_file", help="Prefix for output file. Will be a VCF/CSV if variant_file is VCF/CSV.")
    parser.add_argument("-c", "--column_ids", default="CHROM,POS,REF,ALT", help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
                                                                                "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)")
    parser.add_argument("-m", "--mask", default="True", choices=["False","True"], help="If True, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will be set to 0. (Default: True)")
    parser.add_argument("-s", "--score_cutoff", type=float, help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.")
    parser.add_argument("-d", "--distance", type=int, default=50, help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)")
    parser.add_argument("--score_exons", default="False", choices=["False","True"], help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...")
    parser.add_argument("-b", "--batch_size", type=int, default=3, help="Batch size. (Default: 3)")
    args = parser.parse_args()
  
    batch_size = args.batch_size
    variants = args.variant_file
    gtf = args.annotation_file
    d = args.distance
    fasta = pyfastx.Fasta(args.reference_file)

    try:
        gtf = gffutils.FeatureDB(gtf)
    except:
        print("ERROR, annotation_file could not be opened. Is it a gffutils database file?")
        exit()

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    models = []
    for i in [0,2,4,6]:
        for j in range(1,4):
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)))
            else:
                weights = torch.load(resource_filename(__name__,"models/final.%s.%s.3.v2" % (j, i)), map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)

    if variants.endswith(".vcf"):
        lnum = 0
        # count the number of header lines
        for line in open(variants, 'r'):
            lnum += 1
            if line[0] != '#':
                break

        variants = vcf.Reader(filename=variants)
        variants.infos["Pangolin"] = vcf.parser._Info(
            "Pangolin",'.',"String","Pangolin splice scores. "
            "Format: gene|pos:score_change|pos:score_change|warnings,...",'.','.')
        fout = vcf.Writer(open(args.output_file, 'w'), variants)

        batch_variants = []
        batch_positions = []
        batch_refs = []
        batch_alts = []
        batch_genes_pos = []
        batch_genes_neg = []
        for i, variant in enumerate(variants):
            chr = str(variant.CHROM)
            pos = int(variant.POS)
            ref = variant.REF
            alt = str(variant.ALT[0])
            
            ref_seq, alt_seq, genes_pos, genes_neg = prepare_variant_for_batch(lnum+i, chr, pos, ref, alt, gtf, fasta, d)

            if ref_seq == -1:
                fout.write_record(variant)
                fout.flush()
                continue
            if len(batch_variants) < batch_size-1:
                batch_variants.append(variant)
                batch_positions.append(pos)
                batch_refs.append(ref_seq)
                batch_alts.append(alt_seq)
                batch_genes_pos.append(genes_pos)
                batch_genes_neg.append(genes_neg)
            else:
                batch_variants.append(variant)
                batch_positions.append(pos)
                batch_refs.append(ref_seq)
                batch_alts.append(alt_seq)
                batch_genes_pos.append(genes_pos)
                batch_genes_neg.append(genes_neg)
                batch_scores = process_batch_variants(batch_positions, batch_refs, batch_alts, batch_genes_pos, batch_genes_neg, gtf, models, args)
                for k in range(len(batch_scores)):
                    variant = batch_variants[k]
                    variant.INFO["Pangolin"] = batch_scores[k]
                    fout.write_record(variant)
                    fout.flush()
                batch_variants = []
                batch_positions = []
                batch_refs = []
                batch_alts = []
                batch_genes_pos = []
                batch_genes_neg = []
                
        if len(batch_variants) > 0:
            batch_scores = process_batch_variants(batch_positions, batch_refs, batch_alts, batch_genes_pos, batch_genes_neg, gtf, models, args)
            for k in range(len(batch_scores)):
                variant = batch_variants[k]
                variant.INFO["Pangolin"] = batch_scores[k]
                fout.write_record(variant)
                fout.flush()

        fout.close()

    elif variants.endswith(".csv"):
        col_ids = args.column_ids.split(',')
        variants = pd.read_csv(variants, header=0)
        fout = open(args.output_file+".csv", 'w')
        fout.write(','.join(variants.columns)+',Pangolin\n')
        fout.flush()

        for lnum, variant in variants.iterrows():
            chr, pos, ref, alt = variant[col_ids]
            ref, alt = ref.upper(), alt.upper()
            scores = process_variant(lnum+1, str(chr), int(pos), ref, alt, gtf, models, args)
            if scores == -1:
                fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+'\n')
            else:
                fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+scores+'\n')
            fout.flush()

        fout.close()

    else:
        print("ERROR, variant_file needs to be a CSV or VCF.")

    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()
