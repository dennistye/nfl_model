document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const matchup = document.getElementById('matchup').value;
    const [home_team, away_team] = matchup.split('_');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `home_team=${home_team}&away_team=${away_team}`
    })
    .then(response => response.json())
    .then(data => {
        
        if (data.error) {
            document.getElementById('result').innerText = data.error;
            document.getElementById('spread').innerText = '';
            document.getElementById('total').innerText = '';
        } else {
            document.getElementById('result').innerText = `${data.away_team} at ${data.home_team} — Home Win Probability: ${(data.probability_home_win * 100).toFixed(1)}%`;
            document.getElementById('spread').innerText = `Home Predicted Spread: ${(data.predicted_spread).toFixed(1)}`;
            document.getElementById('total').innerText = `Home Predicted Total: ${(data.predicted_total).toFixed(1)}`;;
        }
    });
    
});


