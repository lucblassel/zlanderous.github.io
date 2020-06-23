var DataFrame = dfjs.DataFrame
console.log('this is a test');


let filename = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"

const COUNTRIES = [
'Austria', 'Belgium', 'Denmark', 'Finland', 'France',
'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
'Luxembourg', 'Netherlands', 'Norway', 'Portugal',
'Spain', 'Sweden', 'Switzerland', 'United Kingdom'
]

//COLORS = ['#00429d', '#2b57a7', '#426cb0', '#5681b9', '#6997c2', '#7daeca', '#93c4d2', '#abdad9', '#caefdf', '#ffe2ca', '#ffc4b4', '#ffa59e', '#f98689', '#ed6976', '#dd4c65', '#ca2f55', '#b11346', '#93003a']

var COLORS = ["#4ec086",
"#b55fb0",
"#a0ad3b",
"#615fb6",
"#6e9343",
"#b64872",
"#20d8fd",
"#b84c3e",
"#c5863b"]

DataFrame.fromCSV(filename)
    .then(console.log('read file'))
    .then(makeChart);

function makeChart(dataframe) {

  console.log('inside makeChart');

  var joined = formatData(dataframe);
  var datasets = getPlotData(joined, COUNTRIES);

  var chart = new Chart('chart', {
    type: 'line',
    data: {
      labels: joined.listColumns().slice(1),
      datasets: datasets,
    },
    options: {
      steppedLine: 'middle',
      scales: {
        yAxes: [{
          scaleLabel: {
            display: true,
            labelString: 'confirmed cases',
          }
        }],
        xAxes: [{
          scaleLabel: {
            display: true,
            labelString: 'date',
          }
        }],
      },
      legend: {
        position: 'bottom',
        labels: {
          boxWidth: 20,
          fontsize: 10,
          align: 'start',
	}
      }
    },
  });

}

function formatData(dataframe) {
  console.log('inside formatData');
  var cols = dataframe.listColumns();

  for (i=0; i<4; i++) {
    cols.shift();
  }

  var summed_columns = [];
  var grouped = dataframe.groupBy('Country/Region');
  for (col of cols) {
    summed_columns.push(
      grouped
        .aggregate(group => group.stat.sum(col))
        .rename('aggregation', col)
    );
  }

  var df = summed_columns.shift().transpose();
  for (df_ of summed_columns) {
    df = df.union(df_.transpose());
  }


  cols.unshift('Country')
  var joined = df.dropDuplicates()
                 .transpose()
                 .renameAll(cols);

  return joined;
}

function getPlotData(joined, countries) {
  console.log('inside getPlotData');
  var datasets = [];
  for (country of countries) {
    row = joined.filter({'Country': country});
    color = COLORS.shift();
    transparent_color = color + '33';
    COLORS.push(color);
    datasets.push(
      {
        label: country,
        data: row.toArray()[0].slice(1),
        borderColor: color,
        backgroundColor: transparent_color,
        pointBackgroundColor: transparent_color,
        pointHoverBackgroundColor: transparent_color,
      }
    );
  };
  return datasets;
}
