from jinja2 import Template
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def extract_taxa_from_nexus(nexus_file):
    taxa = []
    with open(nexus_file, 'r') as file:
        in_taxlabels = False
        for line in file:
            line = line.strip()
            if line.lower().startswith("begin taxa;"):
                in_taxlabels = True
            elif line.lower().startswith("end;"):
                in_taxlabels = False
            elif in_taxlabels and line.lower().startswith("taxlabels"):
                continue
            elif in_taxlabels and line.lower().startswith("end;"):
                break
            elif in_taxlabels and not line.strip().endswith(";"):
                taxa.append(line.strip("'"))
    return taxa

def compute_transition_matrices_and_rewards(location_codes):
    matrices = {}
    rewards = {}

    df = pd.DataFrame(np.zeros(shape=(len(location_codes), len(location_codes))), index=location_codes, columns=location_codes)
    total_count = np.ones(df.shape, dtype=int)
    np.fill_diagonal(total_count, 0)
    for site in location_codes:
        # Into matrix
        df.loc[~df.columns.isin([site]), site] = 1
        matrices[f"Into.{site}"] = df.to_numpy().flatten().astype(int).astype(str).tolist()
        df.loc[~df.columns.isin([site]), site] = 0

        # OutOf matrix
        df.loc[site, ~df.columns.isin([site])] = 1
        matrices[f"OutOf.{site}"] = df.to_numpy().flatten().astype(int).astype(str).tolist()
        df.loc[site, ~df.columns.isin([site])] = 0

    matrices[f"Location.total"] = total_count.flatten().astype(int).astype(str).tolist()
    
    for i in range(len(location_codes)):
        for j in range(len(location_codes)):
            if i == j:
                continue
            df.iloc[i, j] = 1
            matrices[f"Location.{location_codes[i]}.{location_codes[j]}"] = df.to_numpy().flatten().astype(int).astype(str).tolist()
            df.iloc[i, j] = 0

    # Compute rewards matrix
    for i, site in enumerate(location_codes):
        reward_matrix = np.zeros(len(location_codes))
        reward_matrix[i] = 1.0
        rewards[f"rewards_{site}"] = reward_matrix.astype(int).astype(str).tolist()

    return matrices, rewards

def main(input_template, input_trees, output_xml, NAME, chain_length, sample_every):
    with open(input_template, 'r') as template_file:
        template = Template(template_file.read())
        taxa = extract_taxa_from_nexus(input_trees)
        taxa_list = [{'id': taxon.strip(), 'location': taxon.split('|')[1]} for taxon in taxa]
        location_codes = list(set([taxon['location'] for taxon in taxa_list]))
        matrices, rewards = compute_transition_matrices_and_rewards(location_codes)

        rendered_xml = template.render(
            taxa=taxa_list,
            tree_file=Path(input_trees).name,
            log_filename=f"{NAME}.log",
            tree_log_filename=f"{NAME}.trees",
            location_codes=location_codes,
            matrices=matrices,
            rewards=rewards,
            chain_length=chain_length,
            sample_every=sample_every
        )

        with open(output_xml, 'w') as f:
            f.write(rendered_xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_template", help="Input template file path")
    parser.add_argument("--input_trees", help="Input trees file path")
    parser.add_argument("--output_xml", help="Output XML file path")
    parser.add_argument("--name", help="Name of the output files")
    parser.add_argument("--chain_length", help="Chain length")
    parser.add_argument("--sample_every", help="Sample every")
    args = parser.parse_args()

    main(args.input_template, args.input_trees, args.output_xml, args.name, args.chain_length, args.sample_every)