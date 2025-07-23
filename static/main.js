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
        const resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.innerText = data.error;
        } else {
            resultDiv.innerText = `${data.away_team} at ${data.home_team} — Home Win Probability: ${(data.probability_home_win * 100).toFixed(1)}%`;
        }
    });
});
