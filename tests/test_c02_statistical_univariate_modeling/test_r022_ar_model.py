from c01_getting_started import path_to_data
from c02_statistical_univariate_modeling.r022_ar_model import ARModelFMU, plot_predictions_ar_model, \
    train_test_split_consumption, plot_opsd_germany_daily_pacf, read_opsd_germany_daily


def test_r022_ar_model():
    filepath_or_buffer = f'{path_to_data}/input/opsd_germany_daily.csv'
    data = read_opsd_germany_daily(filepath_or_buffer)
    plot_opsd_germany_daily_pacf(data, show=False)
    test_df, train_df, eval_df = train_test_split_consumption(data)
    model_ar = ARModelFMU()
    model_ar.fit(train_df)
    model_ar.read()
    model_ar.save(f"{path_to_data}/output/model_ar.json")
    model_ar2 = ARModelFMU()
    model_ar2.load(f"{path_to_data}/output/model_ar.json")
    model_ar2.read()
    plot_predictions_ar_model(data, model_ar2, test_df, show=False)
    plot_predictions_ar_model(data, model_ar2, eval_df, show=False)
