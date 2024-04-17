#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital.
"""
import argparse

from inflammation import models, views


def main(in_files):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    if not isinstance(in_files, list):
        in_files = [in_files]

    for file_name in in_files:
        inflammation_data = models.load_csv(file_name)

        view_data = {
            'average': models.daily_mean(inflammation_data), 
            'max': models.daily_max(inflammation_data), 
            'min': models.daily_min(inflammation_data),
            'std': models.std_dev(inflammation_data),
        }

        views.visualize(view_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system'
    )

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient'
    )

    args = parser.parse_args()

    main(args.infiles)
