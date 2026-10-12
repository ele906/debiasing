"""Export complete count and SAE CSV tables as readable Markdown."""
import argparse
import csv
from pathlib import Path


def cell(value):
    return str(value).replace('|','\\|').replace('\n','<br>')


def export(path, chunk=6):
    with path.open() as f:
        reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
    if not fields:raise ValueError(f'No header: {path}')
    is_category='category' in fields
    is_race=path.name=='counts_by_race.csv'
    if is_category:
        base=['category','images'];values=[c for c in fields if c not in ['category','race_prompt','gender_prompt','images']]
    elif is_race:
        base=['race_prompt','images'];values=[c for c in fields if c not in base]
    else:
        base=[];values=fields
    groups=[values[i:i+chunk] for i in range(0,len(values),chunk)] if base else [values]
    report=[f'# {path.stem.replace("_"," ").capitalize()}', '', f'Complete table from [{path.name}]({path.name}); {len(rows)} rows. No rows are omitted.', '']
    if path.name.startswith(('counts_','raw_phrase_counts_','rates_')):
        report += ['Values describe unreviewed detector labels in generated images. Categories are generation prompts, not identities inferred from pixels. Cultural clothing and hairstyles are not harm labels. Counts are image-level presence, so repeated boxes do not increase counts.', '']
    if path.name.startswith('rates_'):
        report += ['Rates below are percentages; the CSV stores fractions between 0 and 1.', '']
    for columns in groups:
        headers=base+columns
        report += ['| '+' | '.join(cell(c) for c in headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
        for row in rows:
            cells=[]
            for c in headers:
                value=row[c]
                if path.name.startswith('rates_') and c not in base:
                    value=f'{float(value)*100:.1f}%'
                cells.append(cell(value))
            report.append('| '+' | '.join(cells)+' |')
        report.append('')
    target=path.with_suffix('.md')
    target.write_text('\n'.join(report).rstrip()+'\n')
    print(target,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    names=['counts_by_category.csv','rates_by_category.csv','raw_phrase_counts_by_category.csv','counts_by_race.csv','summary.csv','candidates.csv','intersection_summary.csv','visual_review.csv']
    found=0
    for name in names:
        path=args.directory/name
        if path.exists():export(path);found+=1
    if not found:raise ValueError('No supported tables in directory')


if __name__=='__main__':main()
