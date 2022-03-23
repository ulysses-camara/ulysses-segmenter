/* Setup */
const fetch_url = "http://127.0.0.1:5000/refinery-data-transfer";

const input_uri = "input_content.tsv";

const htmlModTokens = d3.select("#total-modified-tokens");
  
let selectedClass = 1;
let totalModifiedTokens = 0;


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

  fetch(fetch_url, post_response)
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
    return d3.select(this).attr("label") >= 2 ? "red" : "white";
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
    d3.select(this).style("background-color", "#222222");
  });


fn_highlightSelectedClass();

/* Segment section */
fetch(fetch_url)
  .then(function(response) {return response.json(); })
  .then(function(data) {
  const totalTokens = data.length;
  const domSegmentBoard = d3.select(".segment-board");
  const domP = domSegmentBoard.selectAll("span")
    .data(data)
    .text(function(d) { return d["token"]; });
  
  domP.enter().append("span")
    .text(function(d) { return d["token"]; })
    .classed("token", true)
    .classed("hoverable", true)
    .attr("original-label", function(d) { return d["label"]; })
    .attr("label", function(d) { return d3.select(this).attr("original-label"); })
    .attr("token-index", function(_, i) { return i; })
    .on("mouseover", function() {
      const node = d3.select(this);
      if (!node.attr("disabled")) {
        node.style("background-color", "gray");
      }
    })
    .on("mouseout", function() { d3.select(this).style("background-color", "#222222"); })
    .on("click", function() {
      const node = d3.select(this);
  
      if (node.attr("disabled")) {
        return;
      }
  
      const prevLabel = node.attr("label");
  
      if (selectedClass == node.attr("label")) {
        node.attr("label", 0);
      } else {
        node.attr("label", selectedClass);
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
    });
  
  domP.exit().remove();

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
});