var ctx = document.getElementById('myChart').getContext('2d');
        var chart;

        window.onload = function() {
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Random Numbers',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            var socket = io.connect('http://' + document.domain + ':' + location.port);

            socket.on('connect', function() {
                console.log('Connected to server');
            });

            socket.on('new random number', function(data) {
                // 使用当前时间戳作为标签
                var newLabel = new Date().toLocaleTimeString();
                chart.data.labels.push(newLabel);
                chart.data.datasets[0].data.push(data.number);
                if (chart.data.labels.length > 7) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
                chart.update();
            });
        };