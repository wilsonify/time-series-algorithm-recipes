from c01_getting_started.r016_visualization_of_seasonality import boxplot_quarterly_turn_over_data, \
    plot_quarterly_turn_over_data, read_turn_over_data


def test_r016_visualization_of_seasonality():
    turn_over_data = read_turn_over_data()
    quarterly_turn_over_data = plot_quarterly_turn_over_data(turn_over_data, show=False)
    boxplot_quarterly_turn_over_data(quarterly_turn_over_data, show=False)
