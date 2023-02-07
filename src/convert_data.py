import os
import shutil
import pandas as pd

from argparse import ArgumentParser


def main(args):
    from_dir = args.from_dir
    root_dir = args.to_dir
    clips_dir = os.path.join(root_dir, 'clips')

    os.makedirs(clips_dir, exist_ok=True)

    df = pd.DataFrame(columns=['country', 'case_number', 'age_days', 'id', 'path'])

    for f in os.listdir(from_dir):
        from_path = os.path.join(from_dir, f)
        to_path = os.path.join(clips_dir, f)

        country = f[0]
        case_number = f[1:3]
        age_days = f[3:6]
        id = f[6:8]

        if ((country == 'G') or (country == 'J')) and (f[-3:].lower() == 'wav'):
            new_row = {'country': country, 
                       'case_number': case_number,
                       'age_days': age_days,
                       'id': id,
                       'path': to_path}
            
            shutil.copyfile(from_path, to_path)
            
            df = df.append(new_row, ignore_index=True)

    df.to_csv(os.path.join(root_dir, 'babycry.csv'))
    print('done')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--from_dir', type=str, default='/home/tim/Documents/dfki/ds/BabyCry/wavs/')
    parser.add_argument('--to_dir', type=str, default='BabyCry/')

    args = parser.parse_args()
    main(args)