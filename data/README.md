# Experimental data provenance

Source: Vanella et al., *Nature Communications* **15**, 1807 (2024),
[paper](https://doi.org/10.1038/s41467-024-45630-3),
[Zenodo record 8388902](https://doi.org/10.5281/zenodo.8388902).
The downloader pins the record's byte counts and MD5 checksums, verifies cached
files, retries incomplete transfers, and records SHA-256 hashes. It downloads
25.1 MB of processed resources, never the raw Illumina/PacBio archives.

## Exact source and target

Archive: `Figures_data.zip`; workbook: `Figures_data/2309_Figure_2.xlsx`.

| Output | Worksheet | Exact source column | Meaning |
|---|---|---|---|
| `mutation` | `Fig2_H`, `Fig2_F` | `DAOx variant` | Single substitution, one-based RgDAAO numbering |
| `activity` | `Fig2_H` | `activity  score` (two spaces) | Direct consensus EP-Seq activity fitness relative to WT |
| `expression` | `Fig2_F` | `expression score` | Auxiliary expression/folding fitness; never a model input or target |

These are the fitness distributions in paper Fig. 2H and 2F, not the
activity/expression-normalized hotspot scores in Fig. 5/S8. The assay uses
D-alanine on yeast-displayed enzyme. The activity phenotype is a cellular assay
score influenced by expression/folding as well as catalysis, **not purified-enzyme
kcat or a percentage of catalytic activity**.

The paper defines per-replicate fitness as `log2(beta_variant / beta_WT)`
(Eq. 3), then a cell-count-weighted mean over replicates (Eq. 4).
`Activity_fitness.ipynb` cells 3/9 set fluorescence-bin weights to
`min_max_norm(median_fluorescence) + 1`; cells 6/12 compute `logfitness`;
cell 17 weights by `cmt_size`; cell 19 assigns the consensus `activity`.
`Expression_fitness.ipynb` cells 7/13, 18 and 20 implement the corresponding
expression operations. Cell numbers here are **zero-based JSON indices**.
No additional log transform, clipping, scaling, expression division or label
imputation is performed in this repository. The compressed fluorescence scale
means exponentiating a fitness value does not recover a raw enzyme-rate ratio.

## Filtering and counts

1. Read both worksheets; validate identifiers, class tags and finite values.
2. Trim only surrounding source-identifier whitespace. Keep `tag == missense`;
   explicitly exclude synonymous and nonsense substitutions.
3. Inner-join the assay tables on mutation with one-to-one join validation.
4. Check identities and both labels against
   `Figures_data/2309_Figure_S6.xlsx`, sheet `FigS6_A`, columns
   `WT amino acid`, `position`, `Mut amino acid`, `activity fitness`,
   `expression fitness` (absolute tolerance 1e-12).
5. Validate each mutation against the translated deposited reference, reconstruct
   its sequence and sort lexicographically by mutation before splitting.

| Stage | Activity | Expression |
|---|---:|---:|
| Deposited worksheet rows | 7,036 | 7,069 |
| Synonymous, excluded | 300 | 301 |
| Nonsense, excluded | 332 | 334 |
| Single missense | 6,404 | 6,434 |
| Present only in this assay, excluded | 5 | 35 |
| Paired missense retained | **6,399** | **6,399** |

The paper's count is reproduced exactly, without adding or dropping variants
to force it. No missing/non-finite labels, duplicate mutation IDs, duplicate
sequences, invalid amino acids or reference mismatches occur in the final set.
All sequences contain 365 residues and exactly one substitution at positions
2–365. Missing/invalid records cause errors rather than silent corrections.

The paper reports Phred Q20 filtering, 15-nt barcodes, conversion of reads to
sorted-cell counts and at least 10 total cells for the reported replicate
comparison. Those upstream steps are already reflected in the processed
publication tables; they are not rerun or claimed to be independently reproduced
here. This benchmark reproduces the published processed variant set and labels.

## Reference sequence and numbering

`DAOx_ref.fa` is **2,106 nt of construct DNA**, including barcode and fusion
sequence, not an amino-acid FASTA. The authors'
[Generate_LUT_notebook.ipynb](https://zenodo.org/records/8388902/files/Generate_LUT_notebook.ipynb?download=1)
cell 0 annotates Aga2 start nt 600 and DAOx start nt 912 (one-based).
The RgDAAO coding segment is nts **912–2006 inclusive**, immediately after
BamHI (906–911) and before XhoI (2007–2012), consistent with the paper's cloning
method and notebook target positions 2–365. We translate this 1,095-nt segment
with the standard genetic code and exclude fusion/linker/tags.

The fitness notebooks subtract **104** from construct-level mutation positions.
The figure tables already use RgDAAO positions: **no offset is applied again**.
Every retained mutation matches this reference. The complete sequence is stored
in [provenance.json](../docs/audit/provenance.json) and generated `wt.fasta`:

```text
MHSQKRVVVLGSGVIGLSSALILARKGYSVHILARDLPEDVSSQTFASPWAGANWTPFMTLTDGPRQAKWEESTFKKWVELVPTGHAMWLKGTRRFAQNEDGLLGHWYKDITPNYRPLPSSECPPGAIGVTYDTLSVHAPKYCQYLARELQKLGATFERRTVTSLEQAFDGADLVVNATGLGAKSIAGIDDQAAEPIRGQTVLVKSPCKRCTMDSSDPASPAYIIPRPGGEVICGGTYGVGDWDLSVNPETVQRILKHCLRLDPTISSDGTIEGIEVLRHNVGLRPARRGGPRVEAERIVLPLDRTKSPLSLGRGSARAAKEKEVTLVHAYGFSSAGYQQSWGAAEDVAQLVDEAFQRYHGAARE
```

## Deposit audit limitations

The two notebook `input_files/*.csv` files are tab-separated barcode/count
inputs, not the final fitness table. `Look_up_table.tsv` has no header and maps
barcodes to construct-numbered mutations; it is not a phenotype table.

The deposited activity notebook cell 10 has the query
`bin==13 or bin==15 or bin==15 or bin==16`, apparently omitting bin 14. It also
uses hard-coded WT row indices. These are reasons to use the authors' deposited
final measurements instead of silently repairing and re-running upstream
analysis. We cannot establish from this notebook alone whether its typo affected
the published values. Agreement between Fig2 and FigS6 confirms consistency of
the deposit, not independent validation of its raw-data processing.

The notebook code ends at a broader consensus dataframe; it does not fully
encode all later publication-table filtering/aggregation. The benchmark's
provenance is therefore the published figure workbook. This distinction is
explicit so raw-read reproducibility is not overstated.

## Reproduce

```bash
python scripts/download_data.py
python scripts/build_dataset.py
python -m rgdaao.prepare --input data/processed/variants.csv --output data/processed
```

Generated data remain untracked. Compact audit manifests are committed under
`docs/audit/`. The source publication is CC BY 4.0; cite the original authors
when reusing their data. No raw sequencing data are redistributed here.
