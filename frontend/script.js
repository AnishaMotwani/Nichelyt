
"use strict";

/* =====================================================
   CONFIGURATION
===================================================== */

const API_BASE_URL = "http://127.0.0.1:8001";

let opportunityChartInstance = null;
let shapChartInstance = null;
let latestResponse = null;


/* =====================================================
   HELPER FUNCTIONS
===================================================== */

function $(id) {
  return document.getElementById(id);
}


function showToast(message) {
  const toast = $("toast");

  if (!toast) {
    console.log(message);
    return;
  }

  toast.textContent = message;

  toast.hidden = false;
  toast.classList.remove("hidden");

  setTimeout(() => {
    toast.hidden = true;
    toast.classList.add("hidden");
  }, 4000);
}


function escapeHTML(value) {
  return String(value ?? "").replace(
    /[&<>'"]/g,
    (character) => {
      const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;"
      };

      return entities[character];
    }
  );
}


function formatNumber(value, digits = 2) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "N/A";
  }

  return number.toFixed(digits);
}


function findFirst(object, paths, fallback = null) {
  for (const path of paths) {
    const value = path
      .split(".")
      .reduce(
        (current, key) => current?.[key],
        object
      );

    if (value !== undefined && value !== null) {
      return value;
    }
  }

  return fallback;
}


function getChartTextColor() {
  const styles = getComputedStyle(document.body);

  return (
    styles.getPropertyValue("--text-secondary").trim() ||
    "#667085"
  );
}


function getChartGridColor() {
  const styles = getComputedStyle(document.body);

  return (
    styles.getPropertyValue("--border").trim() ||
    "#dce2ec"
  );
}


/* =====================================================
   CHART VALUE LABEL PLUGIN
===================================================== */

/*
  This plugin displays numeric values
  at the end of chart bars.

  It works for:
  1. Vertical bar charts
  2. Horizontal bar charts
*/

const chartValueLabelPlugin = {
  id: "chartValueLabels",

  afterDatasetsDraw(chart) {
    const { ctx } = chart;

    ctx.save();

    chart.data.datasets.forEach(
      (dataset, datasetIndex) => {
        const meta = chart.getDatasetMeta(
          datasetIndex
        );

        meta.data.forEach((bar, index) => {
          const value = Number(
            dataset.data[index]
          );

          if (!Number.isFinite(value)) {
            return;
          }

          const position = bar.tooltipPosition();

          ctx.fillStyle = getChartTextColor();

          ctx.font = "600 11px Arial";

          ctx.textBaseline = "middle";

          const formattedValue = value.toFixed(2);

          /*
            Horizontal bar chart.

            Positive values:
            Label appears to the right.

            Negative values:
            Label appears to the left.
          */

          if (chart.options.indexAxis === "y") {
            const isNegative = value < 0;

            ctx.textAlign = isNegative
              ? "right"
              : "left";

            ctx.fillText(
              formattedValue,

              isNegative
                ? position.x - 8
                : position.x + 8,

              position.y
            );
          }

          /*
            Vertical bar chart.

            Value appears above the bar.
          */

          else {
            ctx.textAlign = "center";

            ctx.fillText(
              formattedValue,

              position.x,

              position.y - 10
            );
          }
        });
      }
    );

    ctx.restore();
  }
};


/* =====================================================
   INITIAL ELEMENTS
===================================================== */

const form = $("analysisForm");

const loadingSection = $("loadingSection");

const resultsSection = $("resultsSection");

const analyzeButton = $("analyzeButton");

const analyzeButtonText = $("analyzeButtonText");

const themeToggle = $("themeToggle");

const apiUrlLabel = $("apiUrlLabel");

const postDate = $("postDate");

const postTime = $("postTime");

const resetButton = $("demoButton");


/* =====================================================
   SECTION VISIBILITY
===================================================== */

function setSectionVisibility(
  element,
  isVisible
) {
  if (!element) {
    return;
  }

  element.hidden = !isVisible;

  element.classList.toggle(
    "hidden",
    !isVisible
  );
}


/* =====================================================
   INITIAL PAGE SETUP
===================================================== */

if (apiUrlLabel) {
  apiUrlLabel.textContent = API_BASE_URL;
}


function getCurrentDate() {
  return new Date()
    .toISOString()
    .slice(0, 10);
}


if (postDate) {
  postDate.value = getCurrentDate();
}


if (postTime) {
  postTime.value = "18:30";
}


/* =====================================================
   LOADING STATE
===================================================== */

function setLoadingState(isLoading) {
  setSectionVisibility(
    loadingSection,
    isLoading
  );

  if (isLoading) {
    setSectionVisibility(
      resultsSection,
      false
    );
  }

  if (analyzeButton) {
    analyzeButton.disabled = isLoading;
  }

  if (analyzeButtonText) {
    analyzeButtonText.textContent = isLoading
      ? "Analysing..."
      : "Analyse Content";
  }
}


/* =====================================================
   FORM DATA
===================================================== */

function getSelectedPostType() {
  const selectedRadio = document.querySelector(
    'input[name="postType"]:checked'
  );

  if (selectedRadio) {
    return selectedRadio.value;
  }

  const selectElement = $("postType");

  return selectElement
    ? selectElement.value
    : "Photo";
}


function getCheckboxValue(
  id,
  defaultValue = false
) {
  const element = $(id);

  if (!element) {
    return defaultValue;
  }

  return element.checked;
}


function getInputValue(
  id,
  fallback = ""
) {
  const element = $(id);

  if (!element) {
    return fallback;
  }

  return element.value.trim();
}


function getPayload() {
  const followersText = getInputValue(
    "followers",
    "0"
  );

  const followersValue = Number(
    followersText
  );

  return {
    niche: getInputValue("niche"),

    goal: getInputValue("goal"),

    post_idea: getInputValue("postIdea"),

    followers: Number.isFinite(followersValue)
      ? followersValue
      : 0,

    post_type: getSelectedPostType(),

    caption: getInputValue("caption"),

    post_date: getInputValue("postDate"),

    post_time: getInputValue("postTime"),

    tone: getInputValue("tone"),

    target_audience: getInputValue(
      "targetAudience",
      "General audience"
    ) || "General audience",

    language: getInputValue(
      "language",
      "English"
    ) || "English",

    run_counterfactual: getCheckboxValue(
      "runCounterfactual",
      true
    ),

    run_shap: getCheckboxValue(
      "runShap",
      true
    ),

    run_gemini: getCheckboxValue(
      "runGemini",
      false
    )
  };
}


/* =====================================================
   FORM VALIDATION
===================================================== */

function validatePayload(payload) {
  if (!payload.niche) {
    showToast("Please enter a niche.");

    return false;
  }

  if (!payload.goal) {
    showToast("Please select a content goal.");

    return false;
  }

  if (!payload.post_idea) {
    showToast("Please enter your content idea.");

    return false;
  }

  if (
    !Number.isFinite(payload.followers) ||
    payload.followers <= 0
  ) {
    showToast(
      "Please enter a valid follower count."
    );

    return false;
  }

  if (!payload.post_type) {
    showToast("Please select a post type.");

    return false;
  }

  return true;
}


/* =====================================================
   METRIC CARDS
===================================================== */

function renderMetricCards(data) {
  const metricCards = $("metricCards");

  if (!metricCards) {
    return;
  }

  const opportunity = findFirst(
    data,
    [
      "opportunity_analysis.summary.average_opportunity_score"
    ]
  );

  const similarity = findFirst(
    data,
    [
      "opportunity_analysis.summary.average_similarity"
    ]
  );

  /*
    Engagement percentile is intentionally
    not displayed.

    It is not included as a summary metric.
  */

  const metrics = [
    {
      label: "Opportunity score",

      value:
        opportunity === null
          ? "N/A"
          : formatNumber(opportunity, 2),

      note:
        "Analytical opportunity indicator"
    },

    {
      label: "Average similarity",

      value:
        similarity === null
          ? "N/A"
          : formatNumber(similarity, 4),

      note:
        "Similarity with relevant content"
    }
  ];

  metricCards.innerHTML = metrics
    .map(
      (metric) => `
        <div class="metric-card">

          <div class="metric-label">
            ${escapeHTML(metric.label)}
          </div>

          <div class="metric-value">
            ${escapeHTML(metric.value)}
          </div>

          <div class="metric-note">
            ${escapeHTML(metric.note)}
          </div>

        </div>
      `
    )
    .join("");
}


/* =====================================================
   OPPORTUNITY CHART
===================================================== */

function renderOpportunityChart(data) {
  const canvas = $("opportunityChart");

  if (
    !canvas ||
    typeof Chart === "undefined"
  ) {
    return;
  }

  if (opportunityChartInstance) {
    opportunityChartInstance.destroy();

    opportunityChartInstance = null;
  }

  const summary = findFirst(
    data,
    [
      "opportunity_analysis.summary"
    ],
    {}
  );

  const saturation = findFirst(
    data,
    [
      "opportunity_analysis.saturation"
    ],
    {}
  );

  const opportunityScore = Number(
    summary.average_opportunity_score || 0
  );

  /*
    Average similarity is generally
    returned by the backend in the
    range of 0 to 1.

    Example:

    0.75 becomes 75

    This converts similarity into
    a percentage-like 0 to 100 scale.
  */

  const rawSimilarity = Number(
    summary.average_similarity || 0
  );

  const similarity = rawSimilarity <= 1
    ? rawSimilarity * 100
    : rawSimilarity;

  const saturationOpportunity = Number(
    saturation.opportunity_from_saturation || 0
  );

  const values = [
    opportunityScore,
    similarity,
    saturationOpportunity
  ];

  opportunityChartInstance = new Chart(
    canvas,
    {
      type: "bar",

      data: {
        labels: [
          "Opportunity score",
          "Average similarity (%)",
          "Saturation opportunity"
        ],

        datasets: [
          {
            label: "Analytical signals",

            data: values,

            backgroundColor: [
              "#6757e8",
              "#20b9a4",
              "#e5a05f"
            ],

            borderRadius: 8,

            borderSkipped: false
          }
        ]
      },

      options: {
        responsive: true,

        maintainAspectRatio: false,

        scales: {
          y: {
            beginAtZero: true,

            ticks: {
              color: getChartTextColor()
            },

            grid: {
              color: getChartGridColor()
            }
          },

          x: {
            ticks: {
              color: getChartTextColor()
            },

            grid: {
              display: false
            }
          }
        },

        plugins: {
          legend: {
            labels: {
              color: getChartTextColor()
            }
          },

          tooltip: {
            callbacks: {
              label(context) {
                const value = Number(
                  context.raw || 0
                );

                return ` ${value.toFixed(2)}`;
              }
            }
          }
        },

        layout: {
          padding: {
            top: 25
          }
        }
      },

      plugins: [
        chartValueLabelPlugin
      ]
    }
  );
}


/* =====================================================
   SHAP SUMMARY
===================================================== */

function renderShapSummary(data) {
  const container = $("shapSummary");

  if (!container) {
    return;
  }

  const contributions = findFirst(
    data,
    [
      "shap_explanation.feature_contributions"
    ],
    []
  );

  if (
    !Array.isArray(contributions) ||
    contributions.length === 0
  ) {
    container.innerHTML = `
      <p class="form-help">
        No SHAP explanation was returned
        by the backend.
      </p>
    `;

    return;
  }

  const rows = contributions
    .slice()
    .sort(
      (a, b) =>
        Math.abs(
          Number(b.contribution || 0)
        ) -
        Math.abs(
          Number(a.contribution || 0)
        )
    )
    .slice(0, 5);

  container.innerHTML = rows
    .map((row) => {
      const feature = String(
        row.feature || "Feature"
      )
        .replace(/^numeric__/, "")
        .replace(/^categorical__/, "");

      const contribution = Number(
        row.contribution || 0
      );

      const direction =
        contribution >= 0
          ? "Positive contribution"
          : "Negative contribution";

      return `
        <div class="content-block">

          <h4>
            ${escapeHTML(feature)}
          </h4>

          <p>
            ${escapeHTML(direction)}:
            ${escapeHTML(
              formatNumber(contribution, 4)
            )}
          </p>

        </div>
      `;
    })
    .join("");
}


/* =====================================================
   SHAP CHART
===================================================== */

function renderShapChart(data) {
  const canvas = $("shapChart");

  if (
    !canvas ||
    typeof Chart === "undefined"
  ) {
    return;
  }

  if (shapChartInstance) {
    shapChartInstance.destroy();

    shapChartInstance = null;
  }

  const contributions = findFirst(
    data,
    [
      "shap_explanation.feature_contributions"
    ],
    []
  );

  /*
    Sort features by absolute contribution.

    Display only the top 5 features.
  */

  const rows = Array.isArray(contributions)
    ? contributions
        .slice()
        .sort(
          (a, b) =>
            Math.abs(
              Number(b.contribution || 0)
            ) -
            Math.abs(
              Number(a.contribution || 0)
            )
        )
        .slice(0, 5)
    : [];

  const labels = rows.map((row) =>
    String(row.feature || "Feature")
      .replace(/^numeric__/, "")
      .replace(/^categorical__/, "")
  );

  const values = rows.map((row) =>
    Number(row.contribution || 0)
  );

  const backgroundColors = values.map(
    (value) =>
      value >= 0
        ? "#20b9a4"
        : "#e58ab5"
  );

  shapChartInstance = new Chart(
    canvas,
    {
      type: "bar",

      data: {
        labels,

        datasets: [
          {
            label: "Feature contribution",

            data: values,

            backgroundColor: backgroundColors,

            borderRadius: 7,

            borderSkipped: false
          }
        ]
      },

      options: {
        indexAxis: "y",

        responsive: true,

        maintainAspectRatio: false,

        layout: {
          padding: {
            right: 55
          }
        },

        scales: {
          x: {
            ticks: {
              color: getChartTextColor()
            },

            grid: {
              color: getChartGridColor()
            }
          },

          y: {
            ticks: {
              color: getChartTextColor()
            },

            grid: {
              display: false
            }
          }
        },

        plugins: {
          legend: {
            labels: {
              color: getChartTextColor()
            }
          },

          tooltip: {
            callbacks: {
              label(context) {
                const value = Number(
                  context.raw || 0
                );

                return ` Contribution: ${
                  value.toFixed(4)
                }`;
              }
            }
          }
        }
      },

      plugins: [
        chartValueLabelPlugin
      ]
    }
  );
}


/* =====================================================
   AI CONTENT
===================================================== */

function renderAIContent(data) {
  const aiContentElement = $("aiContent");

  if (!aiContentElement) {
    return;
  }

  const ai =
    data.gemini_content ||
    data.ai_content ||
    {};

  const fields = [
    ["Hook", ai.hook],

    ["Caption", ai.caption],

    ["Call to action", ai.cta],

    ["Content angle", ai.content_angle],

    ["Rationale", ai.rationale]
  ];

  const contentBlocks = fields
    .filter(([, value]) => value)
    .map(
      ([title, value]) => `
        <div class="content-block">

          <h4>
            ${escapeHTML(title)}
          </h4>

          <p>
            ${escapeHTML(value)}
          </p>

        </div>
      `
    );

  const hashtags = Array.isArray(ai.hashtags)
    ? ai.hashtags
    : [];

  if (hashtags.length > 0) {
    contentBlocks.push(`
      <div class="content-block">

        <h4>
          Hashtags
        </h4>

        <p>
          ${escapeHTML(hashtags.join(" "))}
        </p>

      </div>
    `);
  }

  aiContentElement.innerHTML =
    contentBlocks.length > 0
      ? contentBlocks.join("")
      : `
        <div class="content-block">

          <p>
            No AI content was returned.
          </p>

        </div>
      `;
}


/* =====================================================
   COUNTERFACTUAL RESULTS
===================================================== */

function renderCounterfactualDetails(data) {
  const container = $("counterfactualDetails");

  if (!container) {
    return;
  }

  const counterfactual =
    data.counterfactual_comparison ||
    {};

  const scenarios =
    counterfactual.scenarios ||
    counterfactual.results ||
    counterfactual.comparisons ||
    [];

  if (
    !Array.isArray(scenarios) ||
    scenarios.length === 0
  ) {
    container.innerHTML = `
      <div class="content-block">

        <p>
          No detailed counterfactual scenarios
          were returned by the backend.
        </p>

      </div>
    `;

    return;
  }

  const visibleScenarios = scenarios.slice(0, 5);

  container.innerHTML = visibleScenarios
    .map((scenario, index) => {
      const entries = Object.entries(scenario)
        .slice(0, 6)
        .map(
          ([key, value]) => `
            <p>

              <strong>
                ${escapeHTML(key)}:
              </strong>

              ${escapeHTML(value)}

            </p>
          `
        )
        .join("");

      return `
        <div class="counterfactual-item">

          <h4>
            Scenario ${index + 1}
          </h4>

          ${entries}

        </div>
      `;
    })
    .join("");
}
/* =====================================================
   HIDDEN OPPORTUNITY DISCOVERY
===================================================== */

function renderHiddenOpportunities(data) {
  const container = $("hiddenOpportunityDetails");

  if (!container) {
    return;
  }

  const discovery =
    data.hidden_opportunity_discovery || {};

  const opportunities =
    discovery.opportunities ||
    discovery.suggestions ||
    [];

  if (
        discovery.status !== "success" ||
    !Array.isArray(opportunities) ||
    opportunities.length === 0
  ) {
    container.innerHTML = `
      <div class="content-block">
        <p>
          No hidden opportunities were found for this niche.
        </p>
      </div>
    `;
    return;
  }

  container.innerHTML = opportunities
    .slice(0, 5)
    .map((item, index) => `
      <div class="content-block">
        <h4>
          ${index + 1}. ${escapeHTML(
            item.theme || item.topic || item.opportunity || "Content opportunity"
          )}
        </h4>

        <p>
          ${escapeHTML(
            item.explanation || item.reason || item.description || ""
          )}
        </p>
      </div>
    `)
    .join("");
}


/* =====================================================
   RESULTS RENDERING
===================================================== */

function renderResults(data) {
  latestResponse = data;

  renderMetricCards(data);

  renderOpportunityChart(data);

  renderShapSummary(data);

  renderShapChart(data);

  renderAIContent(data);

  renderCounterfactualDetails(data);
  renderHiddenOpportunities(data);

  setSectionVisibility(
    resultsSection,
    true
  );
}


/* =====================================================
   ANALYSIS FORM SUBMISSION
===================================================== */

if (form) {
  form.addEventListener(
    "submit",
    async (event) => {
      event.preventDefault();

      const payload = getPayload();

      if (!validatePayload(payload)) {
        return;
      }

      setLoadingState(true);

      try {
        const response = await fetch(
          `${API_BASE_URL}/analyze`,
          {
            method: "POST",

            headers: {
              "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
          }
        );

        let data;

        try {
          data = await response.json();
        } catch {
          throw new Error(
            "Backend returned an invalid response."
          );
        }

        if (!response.ok) {
          const backendError =
            data?.detail ||
            `Request failed with status ${response.status}`;

          throw new Error(
            typeof backendError === "string"
              ? backendError
              : JSON.stringify(backendError)
          );
        }

        renderResults(data);

        showToast(
          "Analysis completed successfully."
        );

        if (resultsSection) {
          resultsSection.scrollIntoView({
            behavior: "smooth",

            block: "start"
          });
        }
      } catch (error) {
        console.error(
          "Analysis error:",
          error
        );

        showToast(
          `Analysis failed: ${error.message}`
        );
      } finally {
        setLoadingState(false);
      }
    }
  );
}


/* =====================================================
   RESET FORM
===================================================== */

function resetPostType() {
  const photoRadio = document.querySelector(
    'input[name="postType"][value="Photo"]'
  );

  if (photoRadio) {
    photoRadio.checked = true;
  }
}


function resetCheckboxes() {
  const counterfactual = $(
    "runCounterfactual"
  );

  const shap = $("runShap");

  const gemini = $("runGemini");

  if (counterfactual) {
    counterfactual.checked = true;
  }

  if (shap) {
    shap.checked = true;
  }

  if (gemini) {
    gemini.checked = false;
  }
}


function resetForm() {
  if (!form) {
    return;
  }

  /*
    Reset all native form controls.
  */

  form.reset();


  /*
    Reset text and select fields.
  */

  const niche = $("niche");

  const goal = $("goal");

  const postIdea = $("postIdea");

  const followers = $("followers");

  const postDateInput = $("postDate");

  const postTimeInput = $("postTime");

  const tone = $("tone");

  const language = $("language");

  const targetAudience = $("targetAudience");

  const caption = $("caption");


  if (niche) {
    niche.value = "";
  }

  if (goal) {
    goal.value = "";
  }

  if (postIdea) {
    postIdea.value = "";
  }

  if (followers) {
    followers.value = "";
  }

  if (postDateInput) {
    postDateInput.value = getCurrentDate();
  }

  if (postTimeInput) {
    postTimeInput.value = "18:30";
  }

  if (tone) {
    tone.value = "Educational";
  }

  if (language) {
    language.value = "English";
  }

  if (targetAudience) {
    targetAudience.value = "";
  }

  if (caption) {
    caption.value = "";
  }


  /*
    Reset radio button.
  */

  resetPostType();


  /*
    Reset optional analysis checkboxes.
  */

  resetCheckboxes();


  /*
    Hide previous results.
  */

  setSectionVisibility(
    resultsSection,
    false
  );

  setSectionVisibility(
    loadingSection,
    false
  );


  /*
    Destroy previous charts.
  */

  if (opportunityChartInstance) {
    opportunityChartInstance.destroy();

    opportunityChartInstance = null;
  }

  if (shapChartInstance) {
    shapChartInstance.destroy();

    shapChartInstance = null;
  }


  /*
    Clear old response.
  */

  latestResponse = null;


  /*
    Reset result containers.
  */

  const metricCards = $("metricCards");

  const shapSummary = $("shapSummary");

  const aiContent = $("aiContent");

  const counterfactualDetails = $(
    "counterfactualDetails"
  );


  if (metricCards) {
    metricCards.innerHTML = "";
  }

  if (shapSummary) {
    shapSummary.innerHTML = `
      <p class="form-help">
        SHAP explanation will appear here
        when enabled.
      </p>
    `;
  }

  if (aiContent) {
    aiContent.innerHTML = `
      <p>
        AI-generated suggestions will appear here
        when the option is enabled.
      </p>
    `;
  }

  if (counterfactualDetails) {
    counterfactualDetails.innerHTML = `
      <p class="form-help">
        Counterfactual scenarios will appear here
        when enabled.
      </p>
    `;
  }


  showToast(
    "Form reset successfully."
  );
}


if (resetButton) {
  resetButton.textContent = "Reset Form";

  resetButton.addEventListener(
    "click",
    resetForm
  );
}


/* =====================================================
   THEME TOGGLE
===================================================== */

function updateThemeIcon() {
  if (!themeToggle) {
    return;
  }

  const isDark = document.body.classList.contains(
    "dark"
  );

  const themeIcon = $("themeIcon");

  if (themeIcon) {
    themeIcon.textContent = isDark
      ? "☀"
      : "☾";
  } else {
    themeToggle.textContent = isDark
      ? "☀"
      : "☾";
  }

  themeToggle.setAttribute(
    "aria-label",
    isDark
      ? "Switch to light theme"
      : "Switch to dark theme"
  );
}


if (themeToggle) {
  themeToggle.addEventListener(
    "click",
    () => {
      document.body.classList.toggle(
        "dark"
      );

      document.body.classList.toggle(
        "light"
      );

      updateThemeIcon();

      if (latestResponse) {
        renderOpportunityChart(
          latestResponse
        );

        renderShapChart(
          latestResponse
        );
      }
    }
  );
}


updateThemeIcon();