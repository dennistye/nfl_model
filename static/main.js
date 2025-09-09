const teamLogos = {
  ARI: { color: "#a53e56", logo: "/static/images/ARI.webp" },
  ATL: { color: "#b53e51", logo: "/static/images/ATL.webp" },
  BAL: { color: "#413485", logo: "/static/images/BAL.webp" },
  BUF: { color: "#1e4c9c", logo: "/static/images/BUF.webp" },
  CAR: { color: "#2a98d2", logo: "/static/images/CAR.webp" },
  CHI: { color: "#333b4c", logo: "/static/images/CHI.webp" },
  CIN: { color: "#fc6a3a", logo: "/static/images/CIN.webp" },
  CLE: { color: "#ff5929", logo: "/static/images/CLE.webp" },
  DAL: { color: "#1e3e5c", logo: "/static/images/DAL.webp" },
  DEN: { color: "#1e3e5c", logo: "/static/images/DEN.webp" },
  DET: { color: "#1e88c0", logo: "/static/images/DET.webp" },
  GB: { color: "#3c4e4a", logo: "/static/images/GB.webp" },
  HOU: { color: "#1b292f", logo: "/static/images/HOU.webp" },
  IND: { color: "#1d4671", logo: "/static/images/IND.webp" },
  JAX: { color: "#282828", logo: "/static/images/JAX.webp" },
  KC: { color: "#e83e57", logo: "/static/images/KC.webp" },
  LV: { color: "#282828", logo: "/static/images/LV.webp" },
  LAC: { color: "#1f8fcb", logo: "/static/images/LAC.webp" },
  LA: { color: "#1f4ea2", logo: "/static/images/LA.webp" },
  MIA: { color: "#209da5", logo: "/static/images/MIA.webp" },
  MIN: { color: "#643f92", logo: "/static/images/MIN.webp" },
  NE: { color: "#1e3e5c", logo: "/static/images/NE.webp" },
  NO: { color: "#dcc9a4", logo: "/static/images/NO.webp" },
  NYG: { color: "#293d78", logo: "/static/images/NYG.webp" },
  NYJ: { color: "#296853", logo: "/static/images/NYJ.webp" },
  PHI: { color: "#206369", logo: "/static/images/PHI.webp" },
  PIT: { color: "#282828", logo: "/static/images/PIT.webp" },
  SF: { color: "#b92929", logo: "/static/images/SF.webp" },
  SEA: { color: "#1e3e5c", logo: "/static/images/SEA.webp" },
  TB: { color: "#b23449", logo: "/static/images/TB.webp" },
  TEN: { color: "#1e3e5c", logo: "/static/images/TEN.webp" },
  WAS: { color: "#6a2b2c", logo: "/static/images/WAS.webp" },
  "": { color: "#fff8f2", logo: "/static/images/NYJ.webp" },
};

function loadMatchupData(matchup) {
  // const matchup = document.getElementById('matchup').value;
  const [home_team, away_team] = matchup.split("_");

  // Set team logos
  // Apply home team style
  if (teamLogos[away_team]) {
    document.getElementById("visitor_logo").src = teamLogos[away_team].logo;
    document.querySelector(".team-card:nth-child(1)").style.backgroundColor =
      teamLogos[away_team].color;
  }

  // Apply visitor team style
  if (teamLogos[home_team]) {
    document.getElementById("home_logo").src = teamLogos[home_team].logo;
    document.querySelector(".team-card:nth-child(3)").style.backgroundColor =
      teamLogos[home_team].color;
  }

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `home_team=${home_team}&away_team=${away_team}`,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById(
        "home_spread"
      ).textContent = `Model Spread: ${data.home_spread}`;
      document.getElementById(
        "vegas_home_spread"
      ).textContent = `Vegas Spread: ${data.vegas_home_spread}`;

      document.getElementById(
        "visitor_spread"
      ).textContent = `Model Spread: ${data.visitor_spread}`;
      document.getElementById(
        "vegas_visitor_spread"
      ).textContent = `Vegas Spread: ${data.vegas_visitor_spread}`;

      //   document.getElementById(
      //     "diff_spread"
      //   ).textContent = `Spread Difference: ${data.diff_spread}`;

      document.getElementById("date").textContent = `Date: ${data.date}`;

      document.getElementById("time").textContent = `Time: ${data.time}`;

      document.getElementById(
        "location"
      ).textContent = `Location: ${data.location}`;

      document.getElementById(
        "total"
      ).textContent = `Model Total: ${data.predicted_total}`;
      document.getElementById(
        "vegas_total"
      ).textContent = `Vegas Total: ${data.vegas_total}`;
      //   document.getElementById(
      //     "diff_total"
      //   ).textContent = `Total Difference: ${data.diff_total}`;

      // const spreads = [
      //   // { id: "diff_spread", value: data.diff_spread },
      //   // { id: "diff_total", value: data.diff_total },
      //   { id: "total", value: data.predicted_total },
      //   { id: "vegas_total", value: data.vegas_total },
      // ];

      // spreads.forEach((spread) => {
      //   const element = document.getElementById(spread.id);
      //   if (
      //     spread.value &&
      //     !isNaN(spread.value) &&
      //     Math.abs(spread.value) > 3 &&
      //     spread.id == ("diff_spread" || spread.id == "diff_total")
      //   ) {
      //     element.style.color = "#00f700ff"; // Green for spreads > 3
      //   } else if (spread.id == "total" || spread.id == "vegas_total") {
      //     element.style.color = "black"; // Default color
      //   } else {
      //     element.style.color = "black"; // Default color
      //   }
      // });

      // changing middle table's font to black
      const date = document.getElementById("date");
      date.style.color = "black"; // Default color

      const time = document.getElementById("time");
      time.style.color = "black"; // Default color

      const location = document.getElementById("location");
      location.style.color = "black"; // Default color

      const vegas_total = document.getElementById("vegas_total");
      vegas_total.style.color = "black"; // Default color

      const model_total = document.getElementById("total");
      model_total.style.color = "black"; // Default color

      // --- render starters ---
      function renderStarters(containerId, starters) {
        const container = document.getElementById(containerId);
        container.innerHTML = ""; // Clear previous starters

        const offense = starters.filter((p) => p.side === "offense");
        const defense = starters.filter((p) => p.side === "defense");

        // helper to render a section
        function buildSection(title, players) {
          const section = document.createElement("div");
          section.className = "starter-section";
          section.innerHTML = `<h4>${title}</h4>`;
          const grid = document.createElement("div");
          grid.className = "starters-grid";

          players.sort((a, b) => {
            if (a.position === "QB" && b.position !== "QB") return -1;
            if (a.position !== "QB" && b.position === "QB") return 1;
            return 0;
          });

          players.forEach((p) => {
            const card = document.createElement("div");
            card.className = "starter-card";

            if (p.acquisition === "Injured/Inactive" && p.role === "1 string") {
              card.classList.add("injured");
            }

            const img = document.createElement("img");
            img.src =
              p.headshot ||
              "https://secure.espncdn.com/combiner/i?img=/i/headshots/nophoto.png"; // fallback
            img.alt = p.name;
            img.className = "starter-img";

            const text = document.createElement("div");
            text.textContent = `${p.number} - ${p.name} (${p.position})`;

            card.appendChild(img);
            card.appendChild(text);
            grid.appendChild(card);
          });

          section.appendChild(grid);
          return section;
        }

        // parent flexbox to hold both sections side by side
        const sectionsWrapper = document.createElement("div");
        sectionsWrapper.className = "starters-wrapper";

        sectionsWrapper.appendChild(buildSection("Starting Offense", offense));
        sectionsWrapper.appendChild(buildSection("Starting Defense", defense));

        container.appendChild(sectionsWrapper);
      }

      function renderInjuries(query, starters) {
        const injuryContainer = document.querySelector(query);
        injuryContainer.innerHTML = "";

        let hasInjuries = false;

        starters.forEach((p) => {
          if (p.acquisition === "Injured/Inactive" && p.role === "1 string") {
            // console.log(p.acquisition);

            hasInjuries = true;
            const injuryEntry = document.createElement("div");
            injuryEntry.className = "injury-entry";

            const nameDiv = document.createElement("div");
            nameDiv.textContent = `${p.number} - ${p.name} (${p.position}) (${p.side})`;
            nameDiv.className = "injury-player-name";

            const injuryDiv = document.createElement("div");
            injuryDiv.textContent = p.injury;
            injuryDiv.className = "injury-status";

            // const teamDiv = document.createElement("div");
            // const acronyms = Object.keys(teamLogos); // get all keys
            // injuryDiv.textContent = acronyms[p.team_id - 1];
            // injuryDiv.className = "injury-team";

            injuryEntry.appendChild(nameDiv);
            injuryEntry.appendChild(injuryDiv);
            // injuryEntry.appendChild(teamDiv);
            injuryContainer.appendChild(injuryEntry);
          }
        });
        if (!hasInjuries) {
          const noInjuryDiv = document.createElement("div");
          noInjuryDiv.className = "no-injuries";
          noInjuryDiv.textContent = "No injured players";
          injuryContainer.appendChild(noInjuryDiv);
        }
      }

      renderStarters("home_starters_container", data.home_starters);
      renderStarters("visitor_starters_container", data.visitor_starters);
      renderInjuries(".injury-notification-home", data.home_starters);
      renderInjuries(".injury-notification-visitor", data.visitor_starters);
    });
}

function loadOddsData(odds) {
  const gamesColumn = document.getElementById("odds-column");
  gamesColumn.innerHTML = ""; // Clear previous data

  odds.forEach((game) => {
    // Wrapper for both sections
    const columnItem = document.createElement("div");
    columnItem.className = "column-item";

    const ImageItem = document.createElement("div");
    ImageItem.className = "game-item";

    const logo = document.createElement("img");
    logo.src = teamLogos[game.Visitor].logo; // logo URL
    if (game.Visitor == "NYJ") {
      logo.src =
        "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200";
    }
    logo.alt = game.Visitor;
    logo.className = "team-logo-odds";

    const logo2 = document.createElement("img");
    logo2.src = teamLogos[game.Home].logo; // logo URL
    if (game.Home == "NYJ") {
      logo2.src =
        "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200";
    }
    logo2.alt = game.Home;
    logo2.className = "team-logo-odds";

    ImageItem.appendChild(logo);
    ImageItem.appendChild(logo2);

    // Team names
    const gameItem = document.createElement("div");
    gameItem.className = "game-item";

    const visitor = document.createElement("div");
    visitor.className = "team-cell";
    visitor.textContent = game.Visitor;

    const home = document.createElement("div");
    home.className = "team-cell";
    home.textContent = game.Home;

    gameItem.appendChild(visitor);
    gameItem.appendChild(home);

    // Vegas open
    const VegasOpenoddsItem = document.createElement("div");
    VegasOpenoddsItem.className = "game-item";

    const vegas_open_num1 = document.createElement("div");
    vegas_open_num1.className = "team-cell";
    vegas_open_num1.textContent = game.open_total;

    const vegas_open_num2 = document.createElement("div");
    vegas_open_num2.className = "team-cell";
    vegas_open_num2.textContent = game.open_spread;

    VegasOpenoddsItem.appendChild(vegas_open_num1);
    VegasOpenoddsItem.appendChild(vegas_open_num2);

    // Vegas MoneyLine
    const VegasMLoddsItem = document.createElement("div");
    VegasMLoddsItem.className = "game-item";

    const vegas_visitor_ml = document.createElement("div");
    vegas_visitor_ml.className = "team-cell";
    vegas_visitor_ml.textContent = game.visitor_ml;

    const vegas_home_ml = document.createElement("div");
    vegas_home_ml.className = "team-cell";
    vegas_home_ml.textContent = game.home_ml;

    VegasMLoddsItem.appendChild(vegas_visitor_ml);
    VegasMLoddsItem.appendChild(vegas_home_ml);

    // Vegas Spread
    const VegasSpreadoddsItem = document.createElement("div");
    VegasSpreadoddsItem.className = "game-item";

    const vegas_visitor_spread = document.createElement("div");
    vegas_visitor_spread.className = "team-cell";
    vegas_visitor_spread.textContent = game.visitor_spread;

    const vegas_home_spread = document.createElement("div");
    vegas_home_spread.className = "team-cell";
    vegas_home_spread.textContent = game.home_spread;

    VegasSpreadoddsItem.appendChild(vegas_visitor_spread);
    VegasSpreadoddsItem.appendChild(vegas_home_spread);

    // Vegas Total
    const VegasTotaloddsItem = document.createElement("div");
    VegasTotaloddsItem.className = "game-item";

    const vegas_total_o = document.createElement("div");
    vegas_total_o.className = "team-cell";
    vegas_total_o.textContent = game.o_total;

    const vegas_total_u = document.createElement("div");
    vegas_total_u.className = "team-cell";
    vegas_total_u.textContent = game.u_total;

    VegasTotaloddsItem.appendChild(vegas_total_o);
    VegasTotaloddsItem.appendChild(vegas_total_u);

    // Model Spread
    const ModelSpreadoddsItem = document.createElement("div");
    ModelSpreadoddsItem.className = "game-item";

    const model_visitor_spread = document.createElement("div");
    model_visitor_spread.className = "team-cell";
    model_visitor_spread.textContent = game.VisitorSpread;

    const model_home_spread = document.createElement("div");
    model_home_spread.className = "team-cell";
    model_home_spread.textContent = game.HomeSpread;

    ModelSpreadoddsItem.appendChild(model_visitor_spread);
    ModelSpreadoddsItem.appendChild(model_home_spread);

    // Model Total
    const ModelTotaloddsItem = document.createElement("div");
    ModelTotaloddsItem.className = "game-item";

    const model_total_o = document.createElement("div");
    model_total_o.className = "team-cell";
    model_total_o.textContent = game.o_PredictedTotal;

    const model_total_u = document.createElement("div");
    model_total_u.className = "team-cell";
    model_total_u.textContent = game.u_PredictedTotal;

    ModelTotaloddsItem.appendChild(model_total_o);
    ModelTotaloddsItem.appendChild(model_total_u);

    // Betting Spread

    const BettingSpreadItem = document.createElement("div");
    BettingSpreadItem.className = "game-item";

    const betting_spread1 = document.createElement("div");
    betting_spread1.className = "team-cell";

    const betting_spread2 = document.createElement("div");
    betting_spread2.className = "team-cell";

    if (Math.abs(game.HomeSpread - game.home_spread) > 5) {
      betting_spread2.style.backgroundColor = "lightgreen";
      betting_spread1.textContent = ".";
      betting_spread2.textContent = game.home_spread;
    } else {
      betting_spread1.textContent = ".";
      betting_spread2.textContent = ".";
    }

    BettingSpreadItem.appendChild(betting_spread1);
    BettingSpreadItem.appendChild(betting_spread2);

    // Betting Total

    const BettingTotalItem = document.createElement("div");
    BettingTotalItem.className = "game-item";

    const betting_total1 = document.createElement("div");
    betting_total1.className = "team-cell";

    const betting_total2 = document.createElement("div");
    betting_total2.className = "team-cell";

    let vegas_numericTotal = game.o_total.slice(1);
    if (game.PredictedTotal - vegas_numericTotal < -5) {
      betting_total2.style.backgroundColor = "lightgreen";
      betting_total1.textContent = ".";
      betting_total2.textContent = game.u_total;
    } else if (vegas_numericTotal - game.PredictedTotal < -5) {
      betting_total1.style.backgroundColor = "lightgreen";
      betting_total2.textContent = ".";
      betting_total1.textContent = game.o_total;
    } else {
      betting_total1.textContent = ".";
      betting_total2.textContent = ".";
    }

    BettingTotalItem.appendChild(betting_total1);
    BettingTotalItem.appendChild(betting_total2);

    // Put both side by side inside wrapper
    columnItem.appendChild(ImageItem);
    columnItem.appendChild(gameItem);
    columnItem.appendChild(VegasOpenoddsItem);
    columnItem.appendChild(VegasMLoddsItem);
    columnItem.appendChild(VegasSpreadoddsItem);
    columnItem.appendChild(VegasTotaloddsItem);
    columnItem.appendChild(ModelSpreadoddsItem);
    columnItem.appendChild(ModelTotaloddsItem);
    columnItem.appendChild(BettingSpreadItem);
    columnItem.appendChild(BettingTotalItem);

    // Add to main column
    gamesColumn.appendChild(columnItem);
  });
}

function loadPlayerstats() {}

// Load default matchup on page load
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("odds-column")) {
    fetch("/api/best_odds")
      .then((response) => response.json())
      .then((data) => {
        loadOddsData(data);
      })
      .catch((error) => console.error("Error fetching odds:", error));
  }

  const matchupSelect = document.getElementById("matchup");
  // Set default matchup (e.g., first Week 1 game: PHI vs. DAL)
  const defaultMatchup = "GB_WAS";
  matchupSelect.value = defaultMatchup; // Set dropdown to default
  loadMatchupData(defaultMatchup); // Load data immediately
});

// Update data when matchup changes
document.getElementById("matchup").addEventListener("change", (e) => {
  const matchup = e.target.value;
  if (matchup) {
    loadMatchupData(matchup);
  }
});
