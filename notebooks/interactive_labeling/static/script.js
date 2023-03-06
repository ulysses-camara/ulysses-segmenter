/* Setup */
const FLASK_PORT = 6767;

const fetch_url_data = "http://127.0.0.1:" + FLASK_PORT + "/refinery-data-transfer";
const fetch_url_refresh = "http://127.0.0.1:" + FLASK_PORT + "/call-for-refresh";

const input_uri = "input_content.tsv";

const htmlModTokens = d3.select("#value-total-modified-tokens");

let selectedClass = 1;
let totalModifiedTokens = 0;
let marginHeatmapEnabled = true;
let spaceBarPreviousTimestamp = 0;
const spaceBarDoublePressThreshold = 500;

function fn_eventHandlerKeyup(event) {
  const keyName = event.key.toLowerCase();
  let newSelectedClass = selectedClass;

  if (keyName == "arrowleft" || keyName == "a") {
    newSelectedClass = ((newSelectedClass - 1) % 4 + 4) % 4;
  }

  if (keyName == "arrowright" || keyName == "d") {
    newSelectedClass = (newSelectedClass + 1) % 4;
  }

  if (keyName == "0") { newSelectedClass = 0; }
  if (keyName == "1") { newSelectedClass = 1; }
  if (keyName == "2") { newSelectedClass = 2; }
  if (keyName == "3") { newSelectedClass = 3; }

  if (keyName == " ") {
    const scrollingElement = (document.scrollingElement || document.body);
    const spaceBarCurrentTimestamp = new Date();

    if (spaceBarCurrentTimestamp - spaceBarPreviousTimestamp <= spaceBarDoublePressThreshold) {
      scrollingElement.scrollTop = 0;
      spaceBarCurrentTimestamp = 0;
    } else {
      scrollingElement.scrollTop = scrollingElement.scrollHeight;
    }

    spaceBarPreviousTimestamp = spaceBarCurrentTimestamp;
  }

  if (newSelectedClass !== selectedClass) {
    selectedClass = newSelectedClass;
    fn_highlightSelectedClass();
  }
}


document.addEventListener("keyup", fn_eventHandlerKeyup);


function fn_saveModifications() {
  d3.selectAll(".hoverable")
    .attr("disabled", true)
    .classed("hoverable", false);

  const htmlAllTokens = d3.selectAll(".token");
  const tokensData = htmlAllTokens.data();
  const buttonSend = d3.select(this);

  htmlAllTokens
    .style("border-style", "none");

  d3.selectAll(".label-box, .label-box-commands")
    .style("color", "#666666");

  document.removeEventListener("keyup", fn_eventHandlerKeyup);

  for (let i = 0; i < tokensData.length; i++) {
    const node = d3.select(htmlAllTokens.nodes()[i]);
    tokensData[i]["modified"] = node.attr("label") != node.attr("original-label");
    tokensData[i]["label"] = node.attr("label");
  }

  const post_response = {
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
    method: "POST",
    body: JSON.stringify(tokensData),
  };

  fetch(fetch_url_data, post_response)
    .then(function (response) { return response.status; })
    .then(function (status_code) {
      if (status_code == 200) {
        buttonSend.text("Done. Check Python API for results.");
      } else {
        buttonSend.text("Something went wrong.")
      }
    })
    .catch(error => buttonSend.text("Something went wrong."));
}


function fn_setTokenBackgroundColor() {
  const node = d3.select(this);
  if (marginHeatmapEnabled && node.attr("original-label") != -100 && node.attr("margin")) {
    const margin = +node.attr("margin");
    const redValue =  (1 - margin) * 255 + margin * 34;
    node.style("background-color", "rgb(" + redValue + ", 34, 34)");
  } else {
    node.style("background-color", "#232526");
  }
  return node.style("background-color");
}


function fn_highlightSelectedClass() {
  /* Label section */
  d3.selectAll(".label-box")
    .each(function(d, i) {
      const elem = d3.select(this);
      const clsId = elem.attr("cls-id");
      elem.classed("highlighted-label-box", selectedClass == clsId);
    });
}


function fn_updateTotalModifiedTokens(totalTokens) {
  htmlModTokens.text(
    totalModifiedTokens + " of " + totalTokens +
    " (" + Math.round(10000 * totalModifiedTokens / totalTokens) * 0.01 + "%)"
  );
}


function fn_setTokensTextColor(tokens) {
  if (tokens == undefined) {
    tokens = d3.selectAll(".token");
  }

  tokens.style("color", function() {
    return d3.select(this).attr("label") >= 2 ? "rgb(180, 0, 255)" : "white";
  });
}


function fn_updateSegmentNumbering() {
  d3.selectAll(".segment-start-numbering")
    .text(function(d, i) { return (1 + i) + ". ";});
}


function fn_insertSegmentStartElements(elements) {
  let newElements = elements.select(function() {
      const par = this.parentNode;
      let newHtmlElement = document.createElement("span");
      newHtmlElement.id = "segment-start-" + d3.select(this).attr("token-index");
      return par.insertBefore(newHtmlElement, this);
    })
    .classed("segment-start", true);

  newElements
    .append("hr")
    .classed("segment-start-rule", true);

  newElements
    .append("span")
    .classed("segment-start-numbering", true);
}

function fn_setTokenNewLabel(node, newLabel, totalTokens) {
  if (node.attr("disabled")) { return; }

  const prevLabel = node.attr("label");

  if (newLabel == node.attr("label")) {
    node.attr("label", 0);
  } else {
    node.attr("label", newLabel);
  }

  if (prevLabel == 1 && node.attr("label") != 1) {
    d3.select("#segment-start-" + node.attr("token-index")).remove();
  }

  if (prevLabel != 1 && node.attr("label") == 1) {
    fn_insertSegmentStartElements(node);
  }

  if (prevLabel == 1 || node.attr("label") == 1) {
    fn_updateSegmentNumbering();
  }

  if (prevLabel != node.attr("label")) {
    if (node.attr("label") == node.attr("original-label")) {
      totalModifiedTokens -= 1;
    } else if (prevLabel == node.attr("original-label")) {
      totalModifiedTokens += 1;
    }

    fn_updateTotalModifiedTokens(totalTokens);
  }

  if (node.attr("label") == node.attr("original-label")) {
    node
      .style("border-color", "#CCCCCC")
      .style("border-width", "thin")
      .style("border-style", "dotted");
  } else {
    node
      .style("border-color", "#FCF65E")
      .style("border-style", "solid")
      .style("border-width", "1.5px");
  }

  fn_setTokensTextColor(node);
}

d3.selectAll(".label-box")
  .on("click", function() {
    const node = d3.select(this);
    if (!node.attr("disabled")) {
      selectedClass = node.attr("cls-id");
      fn_highlightSelectedClass();
    }
  });


d3.selectAll("#box-label-prev")
  .on("click", function() {
    if (!d3.select(this).attr("disabled")) {
      selectedClass = ((selectedClass - 1) % 4 + 4) % 4;
      fn_highlightSelectedClass();
    }
  });


d3.selectAll("#box-label-next")
  .on("click", function() {
    if (!d3.select(this).attr("disabled")) {
      selectedClass = (selectedClass + 1) % 4;
      fn_highlightSelectedClass();
    }
  });


d3.selectAll(".label-box, .label-box-commands")
  .on("mouseover", function() {
    const node = d3.select(this);
    if (!node.attr("disabled")) {
      node.style("background-color", "gray");
    }
  })
  .on("mouseout", function() {
    d3.select(this).style("background-color", "#232526");
  });

fn_highlightSelectedClass();

/* Segment section */
fetch(fetch_url_data)
  .then(function(response) {return response.json(); })
  .then(function(data) {
  const totalTokens = data.length;
  const domSegmentBoard = d3.select("#segment-board");
  const domP = domSegmentBoard.selectAll("span")
    .data(data);

  domP.enter().append("span")
    .text(function(d) { return d["token"].replace("##", "᠊"); })
    .classed("token", true)
    .classed("hoverable", true)
    .classed("highlighted", function(d) { return d["highlight"]; })
    .attr("original-label", function(d) { return d["label"]; })
    .attr("margin", function(d) { return d["margin"]; })
    .attr("label", function() { return d3.select(this).attr("original-label"); })
    .attr("token-index", function(_, i) { return i; })
    .style("background-color", fn_setTokenBackgroundColor)
    .on("mouseover", function() {
      const node = d3.select(this);
      if (!node.attr("disabled")) {
        node.style("background-color", "gray");
      }
    })
    .on("mouseout", fn_setTokenBackgroundColor)
    .on("contextmenu", function(e) {
      e.preventDefault();
      const node = d3.select(this);
      fn_setTokenNewLabel(node, node.attr("original-label"), totalTokens)
    })
    .on("click", function() {
      fn_setTokenNewLabel(d3.select(this), selectedClass, totalTokens);
    });

  domP.exit().remove();

  if ("margin" in data[0]) {
    let minMargin = 1.0;
    for (let i = 0; i < data.length; i++) {
      if (data[i]["label"] != -100) {
        minMargin = Math.min(minMargin, +data[i]["margin"]);
      }
    }

    d3.select("#stat-board")
      .select("ul")
      .append("li")
        .text("Minimal margin: ")
        .attr("id", "minimal-margin")
        .append("span")
          .attr("id", "value-minimal-margin")
          .text(Math.round(10000 * minMargin) / 100 + "%");

    const barHeight = 6;
    const barWidth = 320;
    const barLeftShift = 32;

    const svgContainer = d3.select("#minimal-margin")
      .append("svg")
        .attr("id", "svg-minimal-margin")
        .attr("width", barLeftShift + barWidth)
        .attr("height", barHeight);

    svgContainer.append("rect")
      .attr("x", barLeftShift)
      .attr("y", 0)
      .attr("width", barWidth)
      .attr("height", barHeight)
      .style("fill", "white");

    svgContainer.append("rect")
      .attr("x", barLeftShift)
      .attr("y", 0)
      .attr("width", minMargin * barWidth)
      .attr("height", barHeight)
      .style("fill", function() {
        if (minMargin >= 0.70) { return "green"; }
        if (minMargin >= 0.30) { return "orange"; }
        return "red";
      });
  }

  d3.selectAll(".token")
    .filter(function() { return d3.select(this).attr("label") == -100; })
    .style("margin-left", 0)
    .style("padding-left", 0)
    .style("border-left", 0)
    .attr("disabled", true)
    .classed("hoverable", false)
    .style("border-style", "none");

  const htmlSegStartTokens = d3.selectAll(".token")
    .filter(function(d, i) {
      return i == 0 || d3.select(this).attr("label") == 1;
    });

  fn_insertSegmentStartElements(htmlSegStartTokens);

  d3.select("#button-save")
    .on("click", fn_saveModifications);

  /* Setup */
  fn_updateSegmentNumbering();
  fn_setTokensTextColor();
  fn_updateTotalModifiedTokens(totalTokens);

  d3.select("#segment-start-0")
    .selectChild(".segment-start-rule")
      .style("margin-top", "0px")
      .style("margin-bottom", "0px")
      .style("color", "#232526");
});

const intervalInMilliseconds = 5000;
let intervalInSeconds = intervalInMilliseconds * 0.001;
d3.select("#refresh-countdown-panel-value")
  .text(Math.round(intervalInSeconds));

const intervalCountdown = setInterval(function() {
  intervalInSeconds = Math.max(0, intervalInSeconds - 1);
  d3.select("#refresh-countdown-panel-value")
    .text(Math.round(intervalInSeconds));
}, 1000);

const intervalRefresh = setInterval(function() {
  fetch(fetch_url_refresh)
    .then((response) => response.json())
    .then((response_content) => {
      if (response_content["need_refresh"]) {
        window.location.reload();
      }
      intervalInSeconds = intervalInMilliseconds * 0.001;
    });
}, intervalInMilliseconds);
