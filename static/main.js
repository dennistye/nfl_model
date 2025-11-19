const teamLogos = {
  ARI: {
    id: 1,
    color: "#a53e56",
    logo: "/static/images/ARI.webp",
    fullName: "Arizona Cardinals",
  },
  ATL: {
    id: 2,
    color: "#b53e51",
    logo: "/static/images/ATL.webp",
    fullName: "Atlanta Falcons",
  },
  BAL: {
    id: 3,
    color: "#413485",
    logo: "/static/images/BAL.webp",
    fullName: "Baltimore Ravens",
  },
  BUF: {
    id: 4,
    color: "#1e4c9c",
    logo: "/static/images/BUF.webp",
    fullName: "Buffalo Bills",
  },
  CAR: {
    id: 5,
    color: "#2a98d2",
    logo: "/static/images/CAR.webp",
    fullName: "Carolina Panthers",
  },
  CHI: {
    id: 6,
    color: "#333b4c",
    logo: "/static/images/CHI.webp",
    fullName: "Chicago Bears",
  },
  CIN: {
    id: 7,
    color: "#fc6a3a",
    logo: "/static/images/CIN.webp",
    fullName: "Cincinnati Bengals",
  },
  CLE: {
    id: 8,
    color: "#ff5929",
    logo: "/static/images/CLE.webp",
    fullName: "Cleveland Browns",
  },
  DAL: {
    id: 9,
    color: "#1e3e5c",
    logo: "/static/images/DAL.webp",
    fullName: "Dallas Cowboys",
  },
  DEN: {
    id: 10,
    color: "#1e3e5c",
    logo: "/static/images/DEN.webp",
    fullName: "Denver Broncos",
  },
  DET: {
    id: 11,
    color: "#1e88c0",
    logo: "/static/images/DET.webp",
    fullName: "Detroit Lions",
  },
  GB: {
    id: 12,
    color: "#3c4e4a",
    logo: "/static/images/GB.webp",
    fullName: "Green Bay Packers",
  },
  HOU: {
    id: 13,
    color: "#1b292f",
    logo: "/static/images/HOU.webp",
    fullName: "Houston Texans",
  },
  IND: {
    id: 14,
    color: "#1d4671",
    logo: "/static/images/IND.webp",
    fullName: "Indianapolis Colts",
  },
  JAX: {
    id: 15,
    color: "#282828",
    logo: "/static/images/JAX.webp",
    fullName: "Jacksonville Jaguars",
  },
  KC: {
    id: 16,
    color: "#e83e57",
    logo: "/static/images/KC.webp",
    fullName: "Kansas City Chiefs",
  },
  LV: {
    id: 17,
    color: "#282828",
    logo: "/static/images/LV.webp",
    fullName: "Las Vegas Raiders",
  },
  LAC: {
    id: 18,
    color: "#1f8fcb",
    logo: "/static/images/LAC.webp",
    fullName: "Los Angeles Chargers",
  },
  LA: {
    id: 19,
    color: "#1f4ea2",
    logo: "/static/images/LA.webp",
    fullName: "Los Angeles Rams",
  },
  MIA: {
    id: 20,
    color: "#209da5",
    logo: "/static/images/MIA.webp",
    fullName: "Miami Dolphins",
  },
  MIN: {
    id: 21,
    color: "#643f92",
    logo: "/static/images/MIN.webp",
    fullName: "Minnesota Vikings",
  },
  NE: {
    id: 22,
    color: "#1e3e5c",
    logo: "/static/images/NE.webp",
    fullName: "New England Patriots",
  },
  NO: {
    id: 23,
    color: "#dcc9a4",
    logo: "/static/images/NO.webp",
    fullName: "New Orleans Saints",
  },
  NYG: {
    id: 24,
    color: "#293d78",
    logo: "/static/images/NYG.webp",
    fullName: "New York Giants",
  },
  NYJ: {
    id: 25,
    color: "#296853",
    logo: "/static/images/NYJ.webp",
    fullName: "New York Jets",
  },
  PHI: {
    id: 26,
    color: "#206369",
    logo: "/static/images/PHI.webp",
    fullName: "Philadelphia Eagles",
  },
  PIT: {
    id: 27,
    color: "#282828",
    logo: "/static/images/PIT.webp",
    fullName: "Pittsburgh Steelers",
  },
  SF: {
    id: 28,
    color: "#b92929",
    logo: "/static/images/SF.webp",
    fullName: "San Francisco 49ers",
  },
  SEA: {
    id: 29,
    color: "#1e3e5c",
    logo: "/static/images/SEA.webp",
    fullName: "Seattle Seahawks",
  },
  TB: {
    id: 30,
    color: "#b23449",
    logo: "/static/images/TB.webp",
    fullName: "Tampa Bay Buccaneers",
  },
  TEN: {
    id: 31,
    color: "#1e3e5c",
    logo: "/static/images/TEN.webp",
    fullName: "Tennessee Titans",
  },
  WAS: {
    id: 32,
    color: "#6a2b2c",
    logo: "/static/images/WAS.webp",
    fullName: "Washington Commanders",
  },
  "": {
    id: 0,
    color: "#fff8f2",
    logo: "/static/images/NYJ.webp",
    fullName: "Unknown Team",
  },
};

const idToAbbr = Object.entries(teamLogos).reduce((acc, [abbr, team]) => {
  acc[team.id] = abbr;
  return acc;
}, {});

function loadMatchupData(matchup) {
  // const matchup = document.getElementById('matchup').value;
  const [home_team, away_team] = matchup.split("_");

  if (teamLogos[away_team]) {
    document.querySelector(".visitor-title").textContent =
      teamLogos[away_team].fullName;
    document.querySelector(".visitor-title").style.color = "white";
    document.getElementById("visitor_logo").src = teamLogos[away_team].logo;
    document.querySelector(".team-card:nth-child(1)").style.backgroundColor =
      teamLogos[away_team].color;
  }

  if (teamLogos[home_team]) {
    document.querySelector(".home-title").textContent =
      teamLogos[home_team].fullName;
    document.querySelector(".home-title").style.color = "white";
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

      //   ).textContent = `Spread Difference: ${data.diff_spread}`;

      document.getElementById("day").textContent = `Day: ${data.day}`;

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

      // changing middle table's font to black

      const day = document.getElementById("day");
      day.style.color = "black"; // Default color

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
            // console.log(p.injury);
            // console.log(p.name);
            // console.log(p.role);
            if (p.status == "IR" && p.role == "1_string") {
              card.classList.add("injured_reserve");

              const indicator = document.createElement("span");
              indicator.className = "injury-indicator";
              indicator.textContent = "IR";
              card.appendChild(indicator);
            } else if (p.status == "O" && p.role == "1_string") {
              card.classList.add("out");

              const indicator = document.createElement("span");
              indicator.className = "injury-indicator";
              indicator.textContent = "O";
              card.appendChild(indicator);
            } else if (p.status == "Q" && p.role == "1_string") {
              card.classList.add("questionable");

              const indicator = document.createElement("span");
              indicator.className = "injury-indicator";
              indicator.textContent = "Q";
              card.appendChild(indicator);
            } else if (p.status == "D" && p.role == "1_string") {
              card.classList.add("doubtful");

              const indicator = document.createElement("span");
              indicator.className = "injury-indicator";
              indicator.textContent = "D";
              card.appendChild(indicator);
            }

            const img = document.createElement("img");
            img.src =
              p.headshot ||
              "https://secure.espncdn.com/combiner/i?img=/i/headshots/nophoto.png"; // fallback
            img.alt = p.name;
            img.className = "starter-img";

            const text = document.createElement("div");
            text.textContent = `${p.number} - ${p.name} \n(${p.position})`;

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
          console.log(p.role);
          if (p.injury == "Questionable" || p.injury == "Doubtful") {
            // console.log(p.injury);
            // console.log(p.role);
            // console.log(p.acquisition);

            hasInjuries = true;
            const injuryEntry = document.createElement("div");
            injuryEntry.className = "injury-entry";

            const nameDiv = document.createElement("div");
            nameDiv.textContent = `${p.name} (${p.position})`;
            nameDiv.className = "injury-player-name";
            nameDiv.style.fontWeight = "bold";

            const injuryDiv = document.createElement("div");
            const injuryDate = document.createElement("div");
            const injuryDescription = document.createElement("div");

            if (p.injury_description) {
              injuryDiv.textContent = `${p.injury}`;
              injuryDate.textContent = `${p.injury_date}`;
              injuryDescription.textContent = `${p.injury_description}`;
              injuryDiv.className = "injury-status";
              injuryDate.className = "injury-date";
              injuryDescription.className = "injury-description";
            } else {
              injuryDiv.textContent = `${p.injury}`;
              injuryDate.textContent = `${p.injury_date}`;
              injuryDescription.textContent = `No Description`;
              injuryDiv.className = "injury-status";
              injuryDate.className = "injury-date";
              injuryDescription.className = "injury-description";
            }

            // const teamDiv = document.createElement("div");
            // const acronyms = Object.keys(teamLogos); // get all keys
            // injuryDiv.textContent = acronyms[p.team_id - 1];
            // injuryDiv.className = "injury-team";

            injuryEntry.appendChild(nameDiv);
            injuryEntry.appendChild(injuryDiv);
            injuryEntry.appendChild(injuryDate);
            injuryEntry.appendChild(injuryDescription);
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

  odds.sort((a, b) => b.diff_spread - a.diff_spread);

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

    if (game.best_spread == "Home") {
      betting_spread2.style.backgroundColor = "lightgreen";
      betting_spread1.textContent = ".";
      betting_spread1.style.color = "white";
      betting_spread2.textContent = game.home_spread;
    } else if (game.best_spread == "Away") {
      betting_spread1.style.backgroundColor = "lightgreen";
      betting_spread2.textContent = ".";
      betting_spread2.style.color = "white";
      betting_spread1.textContent = game.visitor_spread;
    } else {
      betting_spread1.textContent = ".";
      betting_spread1.style.color = "white";
      betting_spread2.textContent = ".";
      betting_spread2.style.color = "white";
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

    if (game.best_total == "Over") {
      betting_total1.style.backgroundColor = "lightgreen";
      betting_total2.textContent = ".";
      betting_total2.style.color = "white";
      betting_total1.textContent = game.o_total;
    } else if (game.best_total == "Under") {
      betting_total2.style.backgroundColor = "lightgreen";
      betting_total1.textContent = ".";
      betting_total1.style.color = "white";
      betting_total2.textContent = game.u_total;
    } else {
      betting_total1.textContent = ".";
      betting_total1.style.color = "white";
      betting_total2.textContent = ".";
      betting_total2.style.color = "white";
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

function totals_edges(odds) {
  const gamesColumn = document.getElementById("totals_edges");
  gamesColumn.innerHTML = ""; // Clear previous data

  odds.sort((a, b) => b.diff_total - a.diff_total);

  console.log(odds);

  let num = 0;

  odds.forEach((game) => {
    // If we reach here, at least one condition is true — create the card
    const columnItem = document.createElement("div");
    columnItem.className = "column-item";

    // Away team
    const awayDiv = document.createElement("div");
    awayDiv.className = "team-section";
    const awayLogo = document.createElement("img");
    awayLogo.src =
      game.Visitor === "NYJ"
        ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
        : teamLogos[game.Visitor].logo;
    awayLogo.alt = game.Visitor;
    awayLogo.className = "team-logo1";
    const awayName = document.createElement("div");
    awayName.className = "team-name";
    awayName.textContent = game.Visitor;
    awayDiv.appendChild(awayLogo);
    // awayDiv.appendChild(awayName);

    // Home team
    const homeDiv = document.createElement("div");
    homeDiv.className = "team-section";
    const homeLogo = document.createElement("img");
    homeLogo.src =
      game.Home === "NYJ"
        ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
        : teamLogos[game.Home].logo;
    homeLogo.alt = game.Home;
    homeLogo.className = "team-logo1";
    const homeName = document.createElement("div");
    homeName.className = "team-name";
    homeName.textContent = game.Home;
    homeDiv.appendChild(homeLogo);
    // homeDiv.appendChild(homeName);

    // "vs" text
    const vsDiv = document.createElement("div");
    vsDiv.className = "vs-text";
    vsDiv.textContent = "vs";

    // Betting line
    const betDiv = document.createElement("div");
    const edgeDiv = document.createElement("div");

    betDiv.className = "bet-line";
    edgeDiv.className = "edge";

    if (game.best_total == "Over" && num == 0) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && num == 0) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (game.best_total == "Over" && num == 1) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && num == 1) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (game.best_total == "Over" && num == 2) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && num == 2) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else {
      return;
    }

    // Append elements
    columnItem.appendChild(awayDiv);
    columnItem.appendChild(vsDiv);
    columnItem.appendChild(homeDiv);
    columnItem.appendChild(betDiv);
    columnItem.appendChild(edgeDiv);

    gamesColumn.appendChild(columnItem);
    num++;
  });
}

function spread_edges(odds) {
  const gamesColumn = document.getElementById("spreads_edges");
  gamesColumn.innerHTML = ""; // Clear previous data

  odds.sort((a, b) => b.diff_spread - a.diff_spread);

  let num = 0;

  odds.forEach((game) => {
    const columnItem = document.createElement("div");
    columnItem.className = "column-item";

    // Away team
    const awayDiv = document.createElement("div");
    awayDiv.className = "team-section";
    const awayLogo = document.createElement("img");
    awayLogo.src =
      game.Visitor === "NYJ"
        ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
        : teamLogos[game.Visitor].logo;
    awayLogo.alt = game.Visitor;
    awayLogo.className = "team-logo1";
    const awayName = document.createElement("div");
    awayName.className = "team-name";
    awayName.textContent = game.Visitor;
    awayDiv.appendChild(awayLogo);
    // awayDiv.appendChild(awayName);

    // Home team
    const homeDiv = document.createElement("div");
    homeDiv.className = "team-section";
    const homeLogo = document.createElement("img");
    homeLogo.src =
      game.Home === "NYJ"
        ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
        : teamLogos[game.Home].logo;
    homeLogo.alt = game.Home;
    homeLogo.className = "team-logo1";
    const homeName = document.createElement("div");
    homeName.className = "team-name";
    homeName.textContent = game.Home;
    homeDiv.appendChild(homeLogo);
    // homeDiv.appendChild(homeName);

    // "vs" text
    const vsDiv = document.createElement("div");
    vsDiv.className = "vs-text";
    vsDiv.textContent = "vs";

    // Betting line
    const betDiv = document.createElement("div");
    const edgeDiv = document.createElement("div");
    betDiv.className = "bet-line";
    edgeDiv.className = "edge";

    if (num == 0 && game.best_spread == "Home") {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (num == 0 && game.best_spread == "Away") {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (num == 1 && game.best_spread == "Home") {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (num == 1 && game.best_spread == "Away") {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (num == 2 && game.best_spread == "Home") {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else if (num == 2 && game.best_spread == "Away") {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";

      edgeDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
      // betDiv.style.backgroundColor = "lightgreen";
      edgeDiv.style.fontWeight = "bold";
      edgeDiv.style.color = "darkgreen";
    } else {
      return;
    }

    // Append elements
    columnItem.appendChild(awayDiv);
    columnItem.appendChild(vsDiv);
    columnItem.appendChild(homeDiv);
    columnItem.appendChild(betDiv);
    columnItem.appendChild(edgeDiv);

    gamesColumn.appendChild(columnItem);
    num++;
  });
}

function weeks_best_value(data) {
  console.log(data.props);

  const gamesColumn = document.getElementById("weeks_best_value");
  const gamesColumn2 = document.getElementById("weeks_description");
  gamesColumn.innerHTML = ""; // Clear previous data
  gamesColumn2.innerHTML = ""; // Clear previous data

  const maxSpread = Math.max(...data.odds.map((game) => game.diff_spread));
  const maxTotal = Math.max(...data.odds.map((game) => game.diff_total));

  console.log(maxSpread);
  console.log(maxTotal);

  if (maxSpread > maxTotal) {
    data.odds.sort((a, b) => b.diff_spread - a.diff_spread);

    let num = 0;

    data.odds.forEach((game) => {
      const columnItem = document.createElement("div");
      columnItem.className = "column-item";

      // Away team
      const awayDiv = document.createElement("div");
      awayDiv.className = "team-section";
      const awayLogo = document.createElement("img");
      awayLogo.src =
        game.Visitor === "NYJ"
          ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
          : teamLogos[game.Visitor].logo;
      awayLogo.alt = game.Visitor;
      awayLogo.className = "team-logo1";
      awayDiv.appendChild(awayLogo);

      // Home team
      const homeDiv = document.createElement("div");
      homeDiv.className = "team-section";
      const homeLogo = document.createElement("img");
      homeLogo.src =
        game.Home === "NYJ"
          ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
          : teamLogos[game.Home].logo;
      homeLogo.alt = game.Home;
      homeLogo.className = "team-logo1";
      homeDiv.appendChild(homeLogo);

      // "vs" text
      const vsDiv = document.createElement("div");
      vsDiv.className = "vs-text";
      vsDiv.textContent = "vs";

      // Betting line + description
      const betDiv = document.createElement("div");
      betDiv.className = "bet-line";

      const descriptionDiv = document.createElement("div");
      descriptionDiv.className = "bet-description";

      if (game.best_spread == "Home" && num == 0) {
        betDiv.textContent = `${game.Home} ${game.home_spread}`;
        descriptionDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
        betDiv.style.fontWeight = "bold";
        betDiv.style.color = "darkgreen";
        descriptionDiv.style.fontWeight = "bold";
        descriptionDiv.style.color = "darkgreen";
      } else if (game.best_spread == "Away" && num == 0) {
        betDiv.textContent = `${game.Visitor} ${game.visitor_spread}`;
        descriptionDiv.textContent = `EDGE: ${game.spread_percent_edge}%`;
        betDiv.style.fontWeight = "bold";
        betDiv.style.color = "darkgreen";
        descriptionDiv.style.fontWeight = "bold";
        descriptionDiv.style.color = "darkgreen";
      } else {
        return; // Skip if conditions aren’t met
      }

      // Wrap bet + description together
      const betWrapper = document.createElement("div");
      betWrapper.className = "bet-wrapper";
      betWrapper.appendChild(betDiv);
      betWrapper.appendChild(descriptionDiv);

      // Append elements
      columnItem.appendChild(awayDiv);
      columnItem.appendChild(vsDiv);
      columnItem.appendChild(homeDiv);
      // columnItem.appendChild(betWrapper);

      gamesColumn.appendChild(columnItem);
      gamesColumn2.appendChild(betWrapper);

      num++;
    });
  } else {
    data.odds.sort((a, b) => b.diff_total - a.diff_total);

    let num = 0;

    data.odds.forEach((game) => {
      const columnItem = document.createElement("div");
      columnItem.className = "column-item";

      // Away team
      const awayDiv = document.createElement("div");
      awayDiv.className = "team-section";
      const awayLogo = document.createElement("img");
      awayLogo.src =
        game.Visitor === "NYJ"
          ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
          : teamLogos[game.Visitor].logo;
      awayLogo.alt = game.Visitor;
      awayLogo.className = "team-logo1";
      awayDiv.appendChild(awayLogo);

      // Home team
      const homeDiv = document.createElement("div");
      homeDiv.className = "team-section";
      const homeLogo = document.createElement("img");
      homeLogo.src =
        game.Home === "NYJ"
          ? "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200"
          : teamLogos[game.Home].logo;
      homeLogo.alt = game.Home;
      homeLogo.className = "team-logo1";
      homeDiv.appendChild(homeLogo);

      // "vs" text
      const vsDiv = document.createElement("div");
      vsDiv.className = "vs-text";
      vsDiv.textContent = "vs";

      // Betting line + description
      const betDiv = document.createElement("div");
      betDiv.className = "bet-line";

      const descriptionDiv = document.createElement("div");
      descriptionDiv.className = "bet-description";

      if (game.best_total == "Over" && num == 0) {
        betDiv.textContent = `${game.o_total}`;
        descriptionDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
        betDiv.style.fontWeight = "bold";
        betDiv.style.color = "darkgreen";
        descriptionDiv.style.fontWeight = "bold";
        descriptionDiv.style.color = "darkgreen";
      } else if (game.best_total == "Under" && num == 0) {
        betDiv.textContent = `${game.u_total}`;
        descriptionDiv.textContent = `EDGE: ${game.total_percent_edge}%`;
        betDiv.style.fontWeight = "bold";
        betDiv.style.color = "darkgreen";
        descriptionDiv.style.fontWeight = "bold";
        descriptionDiv.style.color = "darkgreen";
      } else {
        return; // Skip if conditions aren’t met
      }

      // Wrap bet + description together
      const betWrapper = document.createElement("div");
      betWrapper.className = "bet-wrapper";
      betWrapper.appendChild(betDiv);
      betWrapper.appendChild(descriptionDiv);

      // Append elements
      columnItem.appendChild(awayDiv);
      columnItem.appendChild(vsDiv);
      columnItem.appendChild(homeDiv);
      // columnItem.appendChild(betWrapper);

      gamesColumn.appendChild(columnItem);
      gamesColumn2.appendChild(betWrapper);

      num++;
    });
  }
}

function best_props(props) {
  const gamesColumn = document.getElementById("props-column");
  gamesColumn.innerHTML = ""; // Clear previous data

  // odds.sort((a, b) => b.diff_spread - a.diff_spread);

  if (!props) {
    gamesColumn.textContent = "No Information At This Time";
  } else {
    let num = 0;
    props.forEach((prop) => {
      if (num == 24) {
        return;
      } else {
        // Wrapper for both sections
        const columnItem = document.createElement("div");
        columnItem.className = "column-item";

        const ImageItem = document.createElement("div");
        ImageItem.className = "game-item";

        // const logo = document.createElement("img");
        // logo.src = teamLogos[game.Visitor].logo; // logo URL
        // if (game.Visitor == "NYJ") {
        //   logo.src =
        //     "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200";
        // }
        // logo.alt = game.Visitor;
        // logo.className = "team-logo-odds";

        // const logo2 = document.createElement("img");
        // logo2.src = teamLogos[game.Home].logo; // logo URL
        // if (game.Home == "NYJ") {
        //   logo2.src =
        //     "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nyj.png&h=200&w=200";
        // }
        // logo2.alt = game.Home;
        // logo2.className = "team-logo-odds";

        // ImageItem.appendChild(logo);
        // ImageItem.appendChild(logo2);

        const container = document.createElement("div");
        container.className = "player-team-container";

        const playerImg = document.createElement("img");
        playerImg.src = prop.player_picture;
        playerImg.alt = prop.full_name;
        playerImg.className = "player-picture";

        const teamImg = document.createElement("img");
        const abbr = idToAbbr[prop.team_id];
        const logo = teamLogos[abbr]?.logo;
        teamImg.src = logo;
        teamImg.alt = abbr || "Team logo";
        teamImg.className = "team-picture";

        container.appendChild(playerImg);
        container.appendChild(teamImg);

        // Vegas open
        const VegasOpenoddsItem = document.createElement("div");
        VegasOpenoddsItem.className = "game-item";

        const vegas_open_num1 = document.createElement("div");
        vegas_open_num1.className = "prop-cell";
        vegas_open_num1.textContent = prop.player_abbr;
        vegas_open_num1.style.fontWeight = "bold";

        const vegas_open_num2 = document.createElement("div");
        vegas_open_num2.className = "prop-cell1";
        vegas_open_num2.textContent = prop.prop_type;

        VegasOpenoddsItem.appendChild(vegas_open_num1);
        VegasOpenoddsItem.appendChild(vegas_open_num2);

        // Vegas MoneyLine
        // const VegasMLoddsItem = document.createElement("div");
        // VegasMLoddsItem.className = "game-item";

        // const vegas_visitor_ml = document.createElement("div");
        // vegas_visitor_ml.className = "prop-cell2";
        // vegas_visitor_ml.textContent = prop.prop_type;

        // const vegas_home_ml = document.createElement("div");
        // vegas_home_ml.className = "team-cell";
        // vegas_home_ml.textContent = game.home_ml;

        // VegasMLoddsItem.appendChild(vegas_visitor_ml);
        // VegasMLoddsItem.appendChild(vegas_home_ml);

        // Vegas Spread
        const VegasSpreadoddsItem = document.createElement("div");
        VegasSpreadoddsItem.className = "game-item";

        const vegas_visitor_spread = document.createElement("div");
        vegas_visitor_spread.className = "team-cell";
        vegas_visitor_spread.textContent = prop.side;

        // const vegas_home_spread = document.createElement("div");
        // vegas_home_spread.className = "team-cell";
        // vegas_home_spread.textContent = game.home_spread;

        VegasSpreadoddsItem.appendChild(vegas_visitor_spread);
        // VegasSpreadoddsItem.appendChild(vegas_home_spread);

        // Vegas Total
        const VegasTotaloddsItem = document.createElement("div");
        VegasTotaloddsItem.className = "game-item";

        const vegas_total_o = document.createElement("div");
        vegas_total_o.className = "team-cell";
        vegas_total_o.textContent = prop.line_value;

        // const vegas_total_u = document.createElement("div");
        // vegas_total_u.className = "team-cell";
        // vegas_total_u.textContent = game.u_total;

        VegasTotaloddsItem.appendChild(vegas_total_o);
        // VegasTotaloddsItem.appendChild(vegas_total_u);

        // Model Spread
        const ModelSpreadoddsItem = document.createElement("div");
        ModelSpreadoddsItem.className = "game-item";

        const model_visitor_spread = document.createElement("div");
        model_visitor_spread.className = "team-cell";
        model_visitor_spread.textContent = prop.odds;

        // const model_home_spread = document.createElement("div");
        // model_home_spread.className = "team-cell";
        // model_home_spread.textContent = game.HomeSpread;

        ModelSpreadoddsItem.appendChild(model_visitor_spread);
        // ModelSpreadoddsItem.appendChild(model_home_spread);

        // Model Total
        const ModelTotaloddsItem = document.createElement("div");
        ModelTotaloddsItem.className = "game-item";

        const model_total_o = document.createElement("div");
        model_total_o.className = "team-cell";
        model_total_o.textContent = Math.round(prop.edge * 100) + "%";

        // const model_total_u = document.createElement("div");
        // model_total_u.className = "team-cell";
        // model_total_u.textContent = game.u_PredictedTotal;

        ModelTotaloddsItem.appendChild(model_total_o);
        // ModelTotaloddsItem.appendChild(model_total_u);

        // Betting Spread

        const BettingSpreadItem = document.createElement("div");
        BettingSpreadItem.className = "game-item";

        const betting_spread1 = document.createElement("div");
        betting_spread1.className = "team-cell";

        const betting_spread2 = document.createElement("div");
        betting_spread2.className = "team-cell";

        betting_spread1.textContent = prop.projection;

        BettingSpreadItem.appendChild(betting_spread1);
        // BettingSpreadItem.appendChild(betting_spread2);

        // // Betting Total

        // const BettingTotalItem = document.createElement("div");
        // BettingTotalItem.className = "game-item";

        // const betting_total1 = document.createElement("div");
        // betting_total1.className = "team-cell";

        // const betting_total2 = document.createElement("div");
        // betting_total2.className = "team-cell";

        // if (game.best_total == "Over") {
        //   betting_total1.style.backgroundColor = "lightgreen";
        //   betting_total2.textContent = ".";
        //   betting_total1.textContent = game.o_total;
        // } else if (game.best_total == "Under") {
        //   betting_total2.style.backgroundColor = "lightgreen";
        //   betting_total1.textContent = ".";
        //   betting_total2.textContent = game.u_total;
        // } else {
        //   betting_total1.textContent = ".";
        //   betting_total2.textContent = ".";
        // }

        // BettingTotalItem.appendChild(betting_total1);
        // BettingTotalItem.appendChild(betting_total2);

        // Put both side by side inside wrapper
        // columnItem.appendChild(ImageItem);
        columnItem.appendChild(container);
        columnItem.appendChild(VegasOpenoddsItem);
        // columnItem.appendChild(VegasMLoddsItem);
        columnItem.appendChild(VegasSpreadoddsItem);
        columnItem.appendChild(VegasTotaloddsItem);
        columnItem.appendChild(ModelSpreadoddsItem);
        columnItem.appendChild(ModelTotaloddsItem);
        columnItem.appendChild(BettingSpreadItem);
        // columnItem.appendChild(BettingTotalItem);

        // Add to main column
        gamesColumn.appendChild(columnItem);

        num++;
      }
    });
  }
}

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
});

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("totals_edges")) {
    fetch("/api/")
      .then((response) => response.json())
      .then((data) => {
        totals_edges(data.odds);
        spread_edges(data.odds);
        weeks_best_value(data);
      })
      .catch((error) => console.error("Error fetching total edges:", error));
  }
});

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("props-column")) {
    fetch("/api/best_props")
      .then((response) => response.json())
      .then((data) => {
        best_props(data);
      })
      .catch((error) => console.error("Error fetching prop edges:", error));
  }
});

// Update data when matchup changes
document.addEventListener("DOMContentLoaded", () => {
  const matchupSelect = document.getElementById("matchup");
  // console.log(matchupSelect);
  if (matchupSelect) {
    // This code will ONLY run on the matchups page
    const defaultMatchup = matchupSelect.options[0].value;
    matchupSelect.value = defaultMatchup;
    loadMatchupData(defaultMatchup);

    matchupSelect.addEventListener("change", (e) => {
      const matchup = e.target.value;
      if (matchup) {
        loadMatchupData(matchup);
        console.log(matchup);
      }
    });
  }
});
