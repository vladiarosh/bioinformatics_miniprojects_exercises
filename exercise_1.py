import requests, sys
import pandas as pd
import mygene


def main():
    input_gene_symbols = ['VEPH1', 'UGDH', 'FRAT2', 'RNU6-686P', 'XYZ']

    # Step 1: Fetch human Ensembl IDs and their chromosome locations
    human_genes_info, human_ensembl_ids = get_gene_details(input_gene_symbols)

    # Step 2: Fetch mouse orthologs, their chromosome locations and gene symbols
    mouse_ortholog_and_gene_info = get_mouse_ortholog(human_ensembl_ids)

    # Step 3: Combine human and mouse data into dataframe and save as CSV
    combined_data = []
    for human, mouse in zip(human_genes_info, mouse_ortholog_and_gene_info):
        human_gene_symbol, human_ensembl_id, human_chromosome = human
        mouse_gene_symbol, mouse_ensembl_id, mouse_chromosome, percent_identity = mouse
        combined_data.append({
            'Human_Gene_Symbol': human_gene_symbol,
            'Human_Ensembl_ID': human_ensembl_id,
            'Human_Chromosome': human_chromosome,
            'Mouse_Gene_Symbol': mouse_gene_symbol,
            'Mouse_Ensembl_ID': mouse_ensembl_id,
            'Mouse_Chromosome': mouse_chromosome,
            'Percent_Identity': percent_identity
        })
    df = pd.DataFrame(combined_data)
    df.to_csv('Orthologs table.csv', index=False)


def get_gene_details(list_of_gene_symbols):
    mg = mygene.MyGeneInfo()
    gene_details = []
    ensembl_ids = []
    for symbol in list_of_gene_symbols:
        gene_info = mg.query(symbol, scopes='symbol', fields='ensembl,genomic_pos', species='human')
        if 'hits' in gene_info and len(gene_info['hits']) > 0:
            ensembl_gene_id = gene_info['hits'][0]['ensembl']['gene']
            genomic_pos = gene_info['hits'][0].get('genomic_pos', {})
            chromosome = genomic_pos.get('chr')
            gene_details.append((symbol, ensembl_gene_id, chromosome))
            ensembl_ids.append(ensembl_gene_id)
        else:
            gene_details.append((f'{symbol} is invalid gene', 'No data', 'No data'))
            ensembl_ids.append('No data')
    return gene_details, ensembl_ids


def get_mouse_ortholog(human_ensembl_ids):
    server = 'https://rest.ensembl.org'
    mg = mygene.MyGeneInfo()
    mouse_data = []
    for human_id in human_ensembl_ids:
        if not human_id:
            mouse_data.append(('No data', 'No data'))
            continue

        ext = f'/homology/id/human/{human_id}?;target_species=mouse'
        response = requests.get(server + ext, headers={"Content-Type": "application/json"})

        if not response.ok:
            response.raise_for_status()
            continue

        result = response.json()
        if 'data' in result and result['data']:
            homologies = result['data'][0]['homologies']
            for homology in homologies:
                if homology['target']['species'] == 'mus_musculus':
                    mouse_id = homology['target']['id']
                    perc_identity = homology['target']['perc_id']

                    # Query MyGene for additional mouse gene details (symbol and chromosome)
                    mouse_gene_info = mg.query(mouse_id, fields='symbol,genomic_pos', species='mouse')
                    if 'hits' in mouse_gene_info and len(mouse_gene_info['hits']) > 0:
                        mouse_symbol = mouse_gene_info['hits'][0].get('symbol')
                        genomic_pos = mouse_gene_info['hits'][0].get('genomic_pos', {})
                        mouse_chromosome = genomic_pos.get('chr')
                    else:
                        mouse_symbol = 'Invalid data or no ortholog'
                        mouse_chromosome = 'Invalid data'
                    mouse_data.append((mouse_symbol, mouse_id, mouse_chromosome, perc_identity))
                    break
                else:
                    mouse_data.append(('No ortholog', 'No data', 'No data', 'No data'))
        else:
            mouse_data.append(('Invalid gene', 'No data', 'No data', 'No data'))
    return mouse_data


if __name__ == "__main__":
    main()





