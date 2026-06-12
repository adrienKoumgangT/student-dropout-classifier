from src.data.load_data import load_raw_data
from src.data.preprocess import get_basic_info, check_data_quality, print_quality_report, inspect_columns


def print_data_info():
    df = load_raw_data('student_dropout_academic_success.csv')

    get_basic_info(df)

    issues = check_data_quality(df)

    print_quality_report(issues)

    summary = inspect_columns(df)
    # print(summary.head().to_string())



if __name__ == '__main__':
    print_data_info()

