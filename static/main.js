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
            // console.log(p.injury);
            // console.log(p.name);
            // console.log(p.role);
            if (
              (p.injury == "Injured reserve" || p.injury == "Out") &&
              p.role == "1 string"
            ) {
              card.classList.add("injured");
            } else if (
              (p.injury == "Questionable" || p.injury == "Doubtful") &&
              p.role == "1 string"
            ) {
              card.classList.add("questionable");
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
          console.log(p.role);
          if (
            p.injury == "Injured reserve" ||
            (p.injury == "Out" && p.role == "1 string")
          ) {
            // console.log(p.injury);
            // console.log(p.role);
            // console.log(p.acquisition);

            hasInjuries = true;
            const injuryEntry = document.createElement("div");
            injuryEntry.className = "injury-entry";

            const nameDiv = document.createElement("div");
            nameDiv.textContent = `${p.number} - ${p.name} (${p.position})`;
            nameDiv.className = "injury-player-name";

            const injuryDiv = document.createElement("div");
            injuryDiv.textContent = `${p.injury} - ${p.injury_date}`;
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

    // if (Math.abs(game.HomeSpread - game.home_spread) > 5) {
    //   betting_spread2.style.backgroundColor = "lightgreen";
    //   betting_spread1.textContent = ".";
    //   betting_spread2.textContent = game.home_spread;
    // } else {
    //   betting_spread1.textContent = ".";
    //   betting_spread2.textContent = ".";
    // }

    console.log(game.best_spread == "Home");
    if (game.best_spread == "Home") {
      betting_spread2.style.backgroundColor = "lightgreen";
      betting_spread1.textContent = ".";
      betting_spread2.textContent = game.home_spread;
    } else if (game.best_spread == "Away") {
      betting_spread1.style.backgroundColor = "lightgreen";
      betting_spread2.textContent = ".";
      betting_spread1.textContent = game.visitor_spread;
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

    // const [vegas_total_diff, vegas_spread_diff] = just_a_good_function(game);
    // console.log(vegas_total_diff);
    // let vegas_numericTotal = game.o_total.slice(1);

    // const u_cond1 = game.PredictedTotal - vegas_numericTotal < -5;
    // const u_cond2 = game.knc === "Under";
    // const u_cond3 = vegas_numericTotal - vegas_total_diff < -1.5; // replace with your third condition

    // const o_cond1 = vegas_numericTotal - game.PredictedTotal < -5;
    // const o_cond2 = game.knc === "Over";
    // const o_cond3 = vegas_numericTotal - vegas_total_diff > 1.5; // replace with your third condition

    // console.log(u_cond3);

    // if ([u_cond1, u_cond2, u_cond3].filter(Boolean).length >= 2) {
    //   betting_total2.style.backgroundColor = "lightgreen";
    //   betting_total1.textContent = ".";
    //   betting_total2.textContent = game.u_total;
    // } else if ([o_cond1, o_cond2, o_cond3].filter(Boolean).length >= 2) {
    //   betting_total1.style.backgroundColor = "lightgreen";
    //   betting_total2.textContent = ".";
    //   betting_total1.textContent = game.o_total;
    // } else {
    //   betting_total1.textContent = ".";
    //   betting_total2.textContent = ".";
    // }

    if (game.best_total == "Over") {
      betting_total1.style.backgroundColor = "lightgreen";
      betting_total2.textContent = ".";
      betting_total1.textContent = game.o_total;
    } else if (game.best_total == "Under") {
      betting_total2.style.backgroundColor = "lightgreen";
      betting_total1.textContent = ".";
      betting_total2.textContent = game.u_total;
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

function totals_edges(odds) {
  const gamesColumn = document.getElementById("totals_edges");
  gamesColumn.innerHTML = ""; // Clear previous data

  odds.forEach((game) => {
    // if (vals > 3) {
    //   return;
    // }
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
    betDiv.className = "bet-line";
    // console.log(game.top_three_total === vals);
    // console.log(game.top_three_total);
    // console.log(vals);
    if (game.best_total == "Over" && game.top_three_total === 1) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_total == "Over" && game.top_three_total === 2) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_total == "Over" && game.top_three_total === 3) {
      betDiv.textContent = game.o_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && game.top_three_total === 1) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && game.top_three_total === 2) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_total == "Under" && game.top_three_total === 3) {
      betDiv.textContent = game.u_total;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else {
      return;
    }

    // Append elements
    columnItem.appendChild(awayDiv);
    columnItem.appendChild(vsDiv);
    columnItem.appendChild(homeDiv);
    columnItem.appendChild(betDiv);

    gamesColumn.appendChild(columnItem);
  });
}

function spread_edges(odds) {
  const gamesColumn = document.getElementById("spreads_edges");
  gamesColumn.innerHTML = ""; // Clear previous data
  let num = 1;
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
    betDiv.className = "bet-line";

    if (game.best_spread == "Home" && game.top_three_spread == 1) {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Away" && game.top_three_spread == 1) {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Home" && game.top_three_spread == 2) {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Away" && game.top_three_spread == 2) {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Home" && game.top_three_spread == 3) {
      betDiv.textContent = game.Home + " " + game.home_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Away" && game.top_three_spread == 3) {
      betDiv.textContent = game.Visitor + " " + game.visitor_spread;
      // betDiv.style.backgroundColor = "lightgreen";
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
    } else {
      return;
    }
    // Append elements
    columnItem.appendChild(awayDiv);
    columnItem.appendChild(vsDiv);
    columnItem.appendChild(homeDiv);
    columnItem.appendChild(betDiv);

    gamesColumn.appendChild(columnItem);
  });
}

function weeks_best_value(odds) {
  const gamesColumn = document.getElementById("weeks_best_value");
  const gamesColumn2 = document.getElementById("weeks_description");
  gamesColumn.innerHTML = ""; // Clear previous data
  gamesColumn2.innerHTML = ""; // Clear previous data

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

    if (game.best_spread == "Home" && game.top_three_spread === 1) {
      betDiv.textContent = `${game.Home} ${game.home_spread}`;
      descriptionDiv.textContent = `MODEL DISCREPANCY BY ${
        Math.round(game.diff_spread * 10) / 10
      } POINTS`;
      betDiv.style.fontWeight = "bold";
      betDiv.style.color = "darkgreen";
      descriptionDiv.style.fontWeight = "bold";
      descriptionDiv.style.color = "darkgreen";
    } else if (game.best_spread == "Away" && game.top_three_spread === 1) {
      betDiv.textContent = `${game.Visitor} ${game.visitor_spread}`;
      descriptionDiv.textContent = `MODEL DISCREPANCY BY ${
        Math.round(game.diff_spread * 10) / 10
      } POINTS`;
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
  });
}

// function just_a_good_function(game) {
//   let cleaned_total = game.open_total;
//   let cleaned_spread = game.open_spread;

//   // Remove 'o' or 'u' from total
//   if (cleaned_total.includes("o") || cleaned_total.includes("u")) {
//     cleaned_total = cleaned_total.replace(/[ou]/gi, "");
//     cleaned_spread = cleaned_spread.replace(/[+-]/g, "");
//   }

//   // Remove '+' or '-' from spread
//   if (cleaned_spread.includes("o") || cleaned_spread.includes("u")) {
//     cleaned_total = cleaned_spread.replace(/[ou]/gi, "");
//     cleaned_spread = cleaned_total.replace(/[+-]/g, "");
//   }

//   return [cleaned_total, cleaned_spread];
// }

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
        totals_edges(data);
        spread_edges(data);
        weeks_best_value(data);
      })
      .catch((error) => console.error("Error fetching total edges:", error));
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
