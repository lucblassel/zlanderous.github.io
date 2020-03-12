
var DataFrame = dfjs.DataFrame

let filename = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"


DataFrame.fromCSV(filename)
    .then(makeChart);

//d3.csv(filename)
//    .then(makeChart);

function makeChart(dataframe) {
  filterData(dataframe);
  var chart = new Chart('chart', {
    type: 'horizontalBar',
    data: {
      labels: ['A', 'B', 'C'],
      datasets: [
        {
          data: [10, 20, 30]
      	}
      ]
    }
  });
}

function filterData(dataframe) {
  var cols = dataframe.listColumns();
  for (i=0; i<4; i++) {
    cols.shift();
  }
  var dfArray = dataframe.groupBy('Country/Region').toCollection();
  for (group of dfArray) {
   // console.log(group);
    if (group['groupKey']['Country/Region'] == 'France') {
      console.log(group['group']['matrix']);
    }
  }
}
