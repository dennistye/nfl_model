
const teamLogos = {
     "ARI": {color: "#a53e56", logo: "/static/images/ARI.webp"},
        "ATL": {color: "#b53e51", logo: "/static/images/ATL.webp"},
        "BAL": {color: "#413485", logo: "/static/images/BAL.webp"},
        "BUF": {color: "#1e4c9c", logo: "/static/images/BUF.webp"},
        "CAR": {color: "#2a98d2", logo: "/static/images/CAR.webp"},
        "CHI": {color: "#333b4c", logo: "/static/images/CHI.webp"},
        "CIN": {color: "#fc6a3a", logo: "/static/images/CIN.webp"},
        "CLE": {color: "#ff5929", logo: "/static/images/CLE.webp"},
        "DAL": {color: "#1e3e5c", logo: "/static/images/DAL.webp"},
        "DEN": {color: "#1e3e5c", logo: "/static/images/DEN.webp"},
        "DET": {color: "#1e88c0", logo: "/static/images/DET.webp"},
        "GB": {color: "#3c4e4a", logo: "/static/images/GB.webp"},
        "HOU": {color: "#1b292f", logo: "/static/images/HOU.webp"},
        "IND": {color: "#1d4671", logo: "/static/images/IND.webp"},
        "JAX": {color: "#282828", logo: "/static/images/JAX.webp"},
        "KC": {color: "#e83e57", logo: "/static/images/KC.webp"},
        "LV": {color: "#282828", logo: "/static/images/LV.webp"},
        "LAC": {color: "#1f8fcb", logo: "/static/images/LAC.webp"},
        "LA": {color: "#1f4ea2", logo: "/static/images/LA.webp"},
        "MIA": {color: "#209da5", logo: "/static/images/MIA.webp"},
        "MIN": {color: "#643f92", logo: "/static/images/MIN.webp"},
        "NE": {color: "#1e3e5c", logo: "/static/images/NE.webp"},
        "NO": {color: "#dcc9a4", logo: "/static/images/NO.webp"},
        "NYG": {color: "#293d78", logo: "/static/images/NYG.webp"},
        "NYJ": {color: "#296853", logo: "/static/images/NYJ.webp"},
        "PHI": {color: "#206369", logo: "/static/images/PHI.webp"},
        "PIT": {color: "#282828", logo: "/static/images/PIT.webp"},
        "SF": {color: "#b92929", logo: "/static/images/SF.webp"},
        "SEA": {color: "#1e3e5c", logo: "/static/images/SEA.webp"},
        "TB": {color: "#b23449", logo: "/static/images/TB.webp"},
        "TEN": {color: "#1e3e5c", logo: "/static/images/TEN.webp"},
        "WAS": {color: "#6a2b2c", logo: "/static/images/WAS.webp"},
        "":{color: "#fff8f2", logo: "/static/images/NYJ.webp"}
}

function loadMatchupData(matchup) {

    // const matchup = document.getElementById('matchup').value;
    const [home_team, away_team] = matchup.split('_');

    // Set team logos
    // Apply home team style
    if (teamLogos[away_team]) {
        document.getElementById("visitor_logo").src = teamLogos[away_team].logo;
        document.querySelector(".team-card:nth-child(1)").style.backgroundColor = teamLogos[away_team].color;
    }

    // Apply visitor team style
    if (teamLogos[home_team]) {
        document.getElementById("home_logo").src = teamLogos[home_team].logo;
        document.querySelector(".team-card:nth-child(2)").style.backgroundColor = teamLogos[home_team].color;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `home_team=${home_team}&away_team=${away_team}`
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("home_spread").textContent = `Model Spread: ${data.home_spread}`;
        document.getElementById("vegas_home_spread").textContent = `Vegas Spread: ${data.vegas_home_spread}`;

        document.getElementById("visitor_spread").textContent = `Model Spread: ${data.visitor_spread}`;
        document.getElementById("vegas_visitor_spread").textContent = `Vegas Spread: ${data.vegas_visitor_spread}`;

        document.getElementById("diff_spread").textContent = `Spread Difference: ${data.diff_spread}`;
        document.getElementById("total").textContent = `Model Total: ${data.predicted_total}`;
        document.getElementById("vegas_total").textContent = `Vegas Total: ${data.vegas_total}`;
        document.getElementById("diff_total").textContent = `Total Difference: ${data.diff_total}`;

         const spreads = [
                    { id: "diff_spread", value: data.diff_spread },
                    { id: "diff_total", value: data.diff_total }
                ];

                spreads.forEach(spread => {
                    const element = document.getElementById(spread.id);
                    if (spread.value && !isNaN(spread.value) && Math.abs(spread.value) > 3) {
                        element.style.color = '#00f700ff'; // Green for spreads > 3
                    } else {
                        element.style.color = 'black'; // Default color
                    }
                });
    });
}

// Load default matchup on page load
document.addEventListener('DOMContentLoaded', () => {
    const matchupSelect = document.getElementById('matchup');
    // Set default matchup (e.g., first Week 1 game: PHI vs. DAL)
    const defaultMatchup = 'PHI_DAL';
    matchupSelect.value = defaultMatchup; // Set dropdown to default
    loadMatchupData(defaultMatchup); // Load data immediately
});

// Update data when matchup changes
document.getElementById('matchup').addEventListener('change', (e) => {
    const matchup = e.target.value;
    if (matchup) {
        loadMatchupData(matchup);
    }
});


