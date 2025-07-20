import json

import numpy as np

from oocr_influence.analysis_utils import InfluenceRunData, TrainingRunData, add_runs_to_run_dict
from oocr_influence.datasets.synthetic_pretraining_docs._call_models import Doc


def get_percentile_sample(df, percentile_bounds: tuple[float, float], n_sample: int, plot_bottom_instead: bool = False):
    """
    Get a random sample of documents within the specified percentile bounds.

    Args:
        df: DataFrame with influence scores
        percentile_bounds: Tuple of (lower_percentile, upper_percentile) between 0-100
        n_sample: Number of documents to sample
        plot_bottom_instead: If True, flip the percentile bounds logic

    Returns:
        DataFrame with sampled documents
    """
    if len(df) == 0:
        return df

    lower_pct, upper_pct = percentile_bounds
    scores = df["influence_score"].values

    # Calculate percentile thresholds
    lower_threshold = np.percentile(scores, lower_pct)
    upper_threshold = np.percentile(scores, upper_pct)

    # Filter documents within percentile bounds
    mask = (scores >= lower_threshold) & (scores <= upper_threshold)
    filtered_df = df[mask]

    # Sample randomly from the filtered set
    if len(filtered_df) == 0:
        return filtered_df

    sample_size = min(n_sample, len(filtered_df))
    sampled_df = filtered_df.sample(n=sample_size, random_state=42)

    # Sort the sample by influence score (descending by default, ascending if plot_bottom_instead)
    sampled_df = sampled_df.sort_values("influence_score", ascending=plot_bottom_instead)

    return sampled_df


def format_document_dropdown(t_item, escape_html_func) -> str:
    """
    Create HTML for a dropdown showing the document field content.
    Returns empty string if document field is None (for pretraining documents).
    """
    document_str = t_item.get("document")
    if document_str is None:
        return ""

    try:
        # Deserialize the Doc pydantic model
        doc = Doc.model_validate_json(document_str)

        # Format the document information
        html_parts = []
        html_parts.append('<details class="document-dropdown">')
        html_parts.append("<summary>View Document Details</summary>")
        html_parts.append('<div class="document-content">')

        # Add key document fields
        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Document Type:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.doc_type)}</div>')
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Document Idea:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.doc_idea)}</div>')
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Reversal Curse:</div>')
        html_parts.append(f'<div class="document-field-value">{doc.reversal_curse}</div>')
        html_parts.append("</div>")

        if doc.additional_text:
            html_parts.append('<div class="document-field">')
            html_parts.append('<div class="document-field-label">Additional Text:</div>')
            html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.additional_text)}</div>')
            html_parts.append("</div>")

        # Show fact information
        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Fact Template ID:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.fact.template.id)}</div>')
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Fact Relation:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.fact.template.relation)}</div>')
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Feature Set Fields:</div>')
        html_parts.append(
            f'<div class="document-field-value">{escape_html_func(str(doc.fact.feature_set.fields))}</div>'
        )
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Universe ID:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.fact.universe_id)}</div>')
        html_parts.append("</div>")

        html_parts.append('<div class="document-field">')
        html_parts.append('<div class="document-field-label">Full Document Text:</div>')
        html_parts.append(f'<div class="document-field-value">{escape_html_func(doc.text)}</div>')
        html_parts.append("</div>")

        html_parts.append("</div>")  # Close document-content
        html_parts.append("</details>")  # Close dropdown

        return "".join(html_parts)

    except Exception as e:
        # If there's an error parsing the document, show the error
        return f'<div class="document-dropdown" style="color: red;">Error parsing document: {escape_html_func(str(e))}</div>'


def output_top_influence_documents_html(
    cdf_extrapolation, # type: ignore # Type of this is found in the influence viz notebook
    run_id_to_data: dict[str, InfluenceRunData | TrainingRunData],
    *,
    query_name: str,
    n_queries: int = 2,
    n_train: int = 20,
    query_ids_to_focus_on: list[str] | None = None,
    plot_bottom_instead: bool = False,
    ids_to_keep: list[str] | None = None,
    group_by_type: bool = True,
    percentile_bounds: tuple[float, float] | None = (20, 80),
    divide_by_gradient_norm: bool = False,
) -> dict[str, str]:
    """
    Build one HTML document per query with colored boxes and scores.
    Returns a dictionary mapping query names to HTML strings.

    Args:
        cdf_extrapolation: CDFExtrapolation object containing name and run_id
        n_queries: Number of queries to process
        n_train: Number of top training documents to show per query (per group if group_by_type=True)
        query_ids_to_focus_on: Optional list of specific query IDs to process
        plot_bottom_instead: If True, show lowest influence scores instead of highest
        ids_to_keep: Optional list of train IDs to include in analysis
        group_by_type: If True, show tabs for each datapoint type with top n_train from each
        percentile_bounds: Optional tuple (lower, upper) percentile bounds for random sample tab
    """

    # Add run to run_dict if not already there
    if cdf_extrapolation.run_id not in run_id_to_data:
        add_runs_to_run_dict(
            [cdf_extrapolation.run_id], run_dict=run_id_to_data, run_type="influence", allow_mismatched_keys=True
        )

    # Get data from the run
    run_data = run_id_to_data[cdf_extrapolation.run_id]
    tokenizer = run_data.tokenizer
    scores_df = run_data.scores_df_dict[query_name]
    train_dataset = run_data.train_dataset_split_by_document
    query_dataset = run_data.test_datasets[query_name]

    # Handle gradient norm data
    gradient_norm_map = None
    has_gradient_norms = False
    if cdf_extrapolation.gradient_norm_run is not None:
        add_runs_to_run_dict(
            [cdf_extrapolation.gradient_norm_run],
            run_dict=run_id_to_data,
            run_type="influence",
            allow_mismatched_keys=True,
        )
        gradient_norm_data = run_id_to_data[cdf_extrapolation.gradient_norm_run]
        gradient_norm_df = gradient_norm_data.scores_df_dict["gradient_norms"]

        # Create a mapping from train_id to gradient norm
        gradient_norm_map = dict(zip(gradient_norm_df["train_id"], gradient_norm_df["influence_score"]))

        # Add gradient norm information to scores_df
        scores_df["gradient_norm"] = scores_df["train_id"].map(gradient_norm_map).fillna(0)

        # Optionally normalize influence scores by dividing by gradient norms
        if divide_by_gradient_norm:

            def normalize_score(row):
                train_id = row["train_id"]
                if train_id in gradient_norm_map:
                    gradient_norm = gradient_norm_map[train_id]
                    if gradient_norm != 0:
                        return row["influence_score"] / (gradient_norm + 0.01)
                return row["influence_score"]

            scores_df["influence_score"] = scores_df.apply(normalize_score, axis=1)

    # For now, prob_vector is None - could be added to CDFExtrapolation if needed
    prob_vector = None

    def escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def get_color_rgb(score: float, std_devs: float) -> tuple[str, str]:
        """Get RGB color string based on standard deviations from mean.
        Returns (background_color, text_color) tuple."""

        # Clamp to reasonable range
        std_devs_clamped = max(-3, min(3, std_devs))

        # Map std devs to intensity (0 to 1)
        intensity = abs(std_devs_clamped) / 3.0

        # Use lighter colors - start from white and add color
        # For positive scores (green tint)
        if score >= 0:
            # Light green: decrease red and blue channels
            r = int(255 - (100 * intensity))  # Reduced from 255 to add green tint
            g = 255  # Keep green at max
            b = int(255 - (100 * intensity))  # Reduced from 255 to add green tint
            bg_color = f"rgb({r}, {g}, {b})"
        else:
            # Light red: decrease green and blue channels
            r = 255  # Keep red at max
            g = int(255 - (100 * intensity))  # Reduced from 255 to add red tint
            b = int(255 - (100 * intensity))  # Reduced from 255 to add red tint
            bg_color = f"rgb({r}, {g}, {b})"

        # Text is always dark for readability
        text_color = "#333"

        return bg_color, text_color

    def check_shared_entity(query_item, train_item) -> bool:
        """Check if query and training item share the same parent entity."""
        # Extract entity information from both items
        # This assumes the entity is stored in the fact structure
        # Adjust the field names based on your actual data structure

        query_entity = None
        train_entity = None

        # Try different possible entity field names
        if "fact" in query_item and query_item["fact"] is not None:
            query_entity = json.loads(query_item["fact"]["fields_json"])["name_of_person"]

        if "fact" in train_item and train_item["fact"] is not None:
            train_entity = json.loads(train_item["fact"]["fields_json"])["name_of_person"]

        # Return True if both entities exist and are the same
        return query_entity is not None and train_entity is not None and query_entity == train_entity

    # Create lookup dictionaries to map IDs to dataset indices
    # Assuming datasets have an 'id' field that matches the IDs in scores_df
    query_id_to_idx = {}
    query_dataset_df = query_dataset.to_pandas()
    for idx, (_, item) in enumerate(query_dataset_df.iterrows()):
        if "id" in item.keys():
            query_id_to_idx[str(item["id"])] = idx
        else:
            # Fallback: if no 'id' field, use index as string
            query_id_to_idx[str(idx)] = idx

    train_id_to_idx = {}
    train_dataset_df = train_dataset.to_pandas()

    if ids_to_keep is not None:
        train_dataset_df = train_dataset_df[train_dataset_df["id"].isin(ids_to_keep)]  # type: ignore
        scores_df = scores_df[scores_df["train_id"].isin(ids_to_keep)]  # type: ignore
    for idx, (_, item) in enumerate(train_dataset_df.iterrows()):
        train_id_to_idx[str(item["id"])] = idx

    # Base CSS styles (updated to include bar chart and enhanced stats)
    base_css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .analysis-title {
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }
        .header {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
            border: 1px solid #e0e0e0;
        }
        
        /* Bar chart styling */
        .bar-chart-container {
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .bar-chart-title {
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 14px;
            color: #333;
        }
        .bar-chart {
            display: flex;
            align-items: flex-end;
            height: 200px;
            border-left: 2px solid #666;
            border-bottom: 2px solid #666;
            padding: 10px 0 0 10px;
            position: relative;
            background: linear-gradient(to top, #f9f9f9 0%, #f9f9f9 100%);
        }
        .bar {
            flex: 1;
            margin: 0 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }
        .bar:hover {
            opacity: 0.8;
            transform: translateY(-2px);
        }
        .bar-fill {
            width: 100%;
            border-radius: 2px 2px 0 0;
            border: 1px solid rgba(0,0,0,0.2);
        }
        .bar-positive {
            background: linear-gradient(to top, #4CAF50, #66BB6A);
        }
        .bar-negative {
            background: linear-gradient(to top, #F44336, #EF5350);
        }
        .bar-label {
            font-size: 10px;
            text-align: center;
            margin-top: 5px;
            color: #666;
            transform: rotate(-45deg);
            transform-origin: center;
            white-space: nowrap;
        }
        .bar-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(0,0,0,0.8);
            color: white;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 12px;
            display: none;
            white-space: nowrap;
            z-index: 1000;
        }
        .bar:hover .bar-tooltip {
            display: block;
        }
        .y-axis-label {
            position: absolute;
            left: -40px;
            top: 50%;
            transform: rotate(-90deg);
            transform-origin: center;
            font-size: 12px;
            color: #666;
        }
        .x-axis-label {
            text-align: center;
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
        
        /* Tab styling */
        .tab-container {
            margin: 20px 0;
        }
        .tab-buttons {
            display: flex;
            border-bottom: 3px solid #e0e0e0;
            margin-bottom: 0;
            flex-wrap: wrap;
        }
        .tab-button {
            background-color: #f5f5f5;
            border: none;
            padding: 12px 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            border-top: 3px solid transparent;
            border-left: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
            transition: all 0.3s ease;
            margin-bottom: -3px;
        }
        .tab-button:first-child {
            border-left: none;
        }
        .tab-button:hover {
            background-color: #e8e8e8;
        }
        .tab-button.active {
            background-color: white;
            border-top: 3px solid #2196F3;
            border-bottom: 3px solid white;
        }
        .tab-content {
            display: none;
            background-color: white;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-top: none;
        }
        .tab-content.active {
            display: block;
        }
        .tab-info {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 14px;
            border-left: 4px solid #2196F3;
        }
        
        .summary-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }
        .summary-table th {
            background-color: #f0f0f0;
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: bold;
        }
        .summary-table td {
            padding: 6px;
            border: 1px solid #ddd;
        }
        .token-section {
            margin-top: 30px;
        }
        .token-header {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .doc-header {
            margin: 15px 0 10px 0;
            font-weight: bold;
            color: #333;
            background-color: #f8f8f8;
            padding: 8px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .doc-type {
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: normal;
        }
        .toggle-button {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.3s;
            margin: 0 5px;
        }
        .toggle-button:hover {
            background-color: #1976D2;
        }
        .view-toggle-container {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .view-toggle-label {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .score-label {
            display: block;
            font-size: 11px;
            margin-top: 2px;
            color: #555;
            text-align: center;
        }
        .token-with-score {
            display: inline-block;
            vertical-align: top;
            margin: 2px;
        }
        .token-display {
            line-height: 2.2;
            margin-bottom: 10px;
            font-family: 'Courier New', monospace;
        }
        .token-display.detailed-view {
            line-height: 1.5;
        }
        .token-cell {
            display: inline-block;
            padding: 4px 6px;
            margin: 2px;
            border-radius: 4px;
            text-align: center;
            min-width: 20px;
            border: 1px solid rgba(0,0,0,0.1);
            position: relative;
            font-size: 14px;
            vertical-align: top;
        }
        .std-dev-indicator {
            position: absolute;
            top: -8px;
            right: -8px;
            font-size: 10px;
            background-color: rgba(0,0,0,0.7);
            color: white;
            padding: 1px 4px;
            border-radius: 10px;
            font-weight: bold;
        }
        .score-display {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            margin-bottom: 20px;
        }
        .positive-score {
            color: #2e7d32;
            font-weight: bold;
        }
        .negative-score {
            color: #c62828;
            font-weight: bold;
        }
        .parent-tag {
            background-color: #4CAF50;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .not-parent-tag {
            background-color: #9e9e9e;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .shared-entity-tag {
            background-color: #FF9800;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .no-shared-entity-tag {
            background-color: #757575;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        .stats-info {
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
            font-style: italic;
        }
        /* Hide score labels by default */
        .token-display .score-label {
            display: none;
        }
        /* Show score labels when in detailed view */
        .token-display.detailed-view .score-label {
            display: block;
        }
        /* Hide std-dev indicators in detailed view */
        .token-display.detailed-view .std-dev-indicator {
            display: none;
        }
        /* Continuous text view styles */
        .token-display.text-view .token-with-score {
            display: inline;
            margin: 0;
        }
        .token-display.text-view .token-cell {
            display: inline;
            padding: 0;
            margin: 0;
            border-radius: 0;
            border: none;
            min-width: auto;
            position: static;
            font-family: inherit;
        }
        .token-display.text-view .score-label {
            display: none;
        }
        .token-display.text-view .std-dev-indicator {
            display: none;
        }
        
        /* Gradient norm view styles */
        .token-display.gradient-norm-view .token-cell {
            /* Override background color for gradient norm view */
            background-color: #FFE0B2 !important;
            color: #E65100 !important;
        }
        
        /* Document dropdown styles */
        .document-dropdown {
            margin: 10px 0;
        }
        .document-dropdown summary {
            cursor: pointer;
            padding: 8px 12px;
            background-color: #f0f8ff;
            border: 1px solid #2196F3;
            border-radius: 4px;
            font-weight: bold;
            color: #1976d2;
            list-style: none;
            transition: background-color 0.3s;
        }
        .document-dropdown summary:hover {
            background-color: #e3f2fd;
        }
        .document-dropdown summary::-webkit-details-marker {
            display: none;
        }
        .document-dropdown summary::before {
            content: "▶ ";
            transition: transform 0.3s;
        }
        .document-dropdown[open] summary::before {
            transform: rotate(90deg);
        }
        .document-content {
            padding: 15px;
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-top: none;
            border-radius: 0 0 4px 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        .document-field {
            margin-bottom: 10px;
        }
        .document-field-label {
            font-weight: bold;
            color: #333;
            font-family: Arial, sans-serif;
        }
        .document-field-value {
            margin-left: 10px;
            color: #666;
        }
    </style>
    <script>
        function toggleScoreView(docId) {
            const container = document.getElementById('tokens-' + docId);
            const button = document.getElementById('toggle-' + docId);
            const isDetailed = container.classList.toggle('detailed-view');
            button.textContent = isDetailed ? 'Hide All Scores' : 'Show All Scores';
        }
        
        function toggleViewMode(queryId, tabSuffix) {
            const allContainers = document.querySelectorAll(`[id*="tokens-doc_${queryId}_"][id*="_${tabSuffix}"]`);
            const button = document.getElementById(`view-toggle-${queryId}-${tabSuffix}`);
            
            let isTextView = false;
            if (allContainers.length > 0) {
                isTextView = allContainers[0].classList.toggle('text-view');
                allContainers.forEach(container => {
                    if (isTextView) {
                        container.classList.add('text-view');
                    } else {
                        container.classList.remove('text-view');
                    }
                });
            }
            
            button.textContent = isTextView ? 'Switch to Token View' : 'Switch to Text View';
        }
        
        function switchTab(queryId, tabName) {
            // Check if this is a sub-tab (contains a hyphen in the middle)
            if (queryId.includes('-') && (tabName === 'top' || tabName === 'sample')) {
                // This is a sub-tab switch
                const parentTabId = queryId;
                
                // Hide all sub-tab contents for this parent
                const allSubContents = document.querySelectorAll(`[id^="tab-content-${parentTabId}-"]`);
                allSubContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all sub-tab buttons for this parent
                const allSubButtons = document.querySelectorAll(`[id^="subtab-button-${parentTabId}-"]`);
                allSubButtons.forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show the selected sub-tab content
                document.getElementById(`tab-content-${parentTabId}-${tabName}`).classList.add('active');
                
                // Mark the selected sub-tab button as active
                document.getElementById(`subtab-button-${parentTabId}-${tabName}`).classList.add('active');
            } else {
                // This is a main tab switch
                // Hide all tab contents for this query
                const allContents = document.querySelectorAll(`[id^="tab-content-${queryId}-"]`);
                allContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all buttons for this query
                const allButtons = document.querySelectorAll(`[id^="tab-button-${queryId}-"]`);
                allButtons.forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show the selected tab content
                document.getElementById(`tab-content-${queryId}-${tabName}`).classList.add('active');
                
                // Mark the selected button as active
                document.getElementById(`tab-button-${queryId}-${tabName}`).classList.add('active');
                
                // If this main tab contains sub-tabs, activate the first sub-tab
                const firstSubTabButton = document.getElementById(`subtab-button-${queryId}-${tabName}-top`);
                if (firstSubTabButton) {
                    firstSubTabButton.click();
                }
            }
        }
        
        function scrollToDocument(docId) {
            const element = document.getElementById(docId);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Highlight the document briefly
                element.style.backgroundColor = '#fff3cd';
                setTimeout(() => {
                    element.style.backgroundColor = '';
                }, 2000);
            }
        }
        
        // Initialize first tab as active for each query when page loads
        document.addEventListener('DOMContentLoaded', function() {
            const firstButtons = document.querySelectorAll('[id$="-tab-0"]');
            firstButtons.forEach(button => {
                if (button.id.includes('tab-button-')) {
                    button.click();
                }
            });
        });
    </script>
    """

    # Get unique query IDs to process
    all_query_ids = scores_df["query_id"].unique()

    if query_ids_to_focus_on is not None:
        query_ids = [qid for qid in query_ids_to_focus_on if qid in all_query_ids][:n_queries]
    else:
        sorted_query_ids = sorted(
            [qid for qid in all_query_ids if str(qid) in query_id_to_idx], key=lambda x: query_id_to_idx[str(x)]
        )

        query_ids = sorted_query_ids[:n_queries]

    html_docs: dict[str, str] = {}

    for query_id in query_ids:
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head>")
        html_parts.append(f"<title>Query {query_id} Influence Analysis - {cdf_extrapolation.name}</title>")
        html_parts.append(base_css)
        html_parts.append("</head><body>")
        html_parts.append('<div class="container">')

        # Add analysis title
        html_parts.append(
            f'<div class="analysis-title">Influence Analysis: {escape_html(cdf_extrapolation.name)}</div>'
        )

        # Get query index from lookup dictionary
        q_idx = query_id_to_idx[str(query_id)]

        # Header
        q_item = query_dataset_df.iloc[q_idx]
        header_text = f"Query {query_id}: PROMPT: {q_item['prompt']} | COMPLETION: {q_item['completion']}"
        if prob_vector is not None and q_idx < len(prob_vector):
            header_text += f" | P(correct)={prob_vector[q_idx]:.3f}"

        html_parts.append(f'<div class="header">{escape_html(header_text)}</div>')

        # Get all influence scores for this query
        query_df = scores_df[scores_df["query_id"] == query_id]

        if group_by_type:
            # Group by datapoint type and get top n_train from each group
            datapoint_types = sorted(query_df["datapoint_type"].unique())

            # Create tabbed interface
            html_parts.append('<div class="tab-container">')
            html_parts.append('<div class="tab-buttons">')

            # Add "All" tab first
            html_parts.append(
                f'<button id="tab-button-{query_id}-all" class="tab-button" onclick="switchTab(\'{query_id}\', \'all\')">All Types</button>'
            )

            # Add tab for each datapoint type
            for i, dtype in enumerate(datapoint_types):
                type_count = len(query_df[query_df["datapoint_type"] == dtype])
                html_parts.append(
                    f'<button id="tab-button-{query_id}-{i}" class="tab-button" onclick="switchTab(\'{query_id}\', \'{i}\')">{escape_html(dtype)} ({type_count})</button>'
                )

            html_parts.append("</div>")

            # Create content for "All" tab
            html_parts.append(f'<div id="tab-content-{query_id}-all" class="tab-content">')
            html_parts.append('<div class="tab-info">Showing top documents across all datapoint types</div>')

            if plot_bottom_instead:
                all_sorted = query_df.nsmallest(n_train, "influence_score")
            else:
                all_sorted = query_df.nlargest(n_train, "influence_score")

            html_parts.extend(
                create_content_for_group(
                    all_sorted,
                    query_id,
                    train_id_to_idx,
                    train_dataset_df,
                    q_item,
                    tokenizer,
                    escape_html,
                    get_color_rgb,
                    check_shared_entity,
                    "all",
                )
            )
            html_parts.append("</div>")

            # Create content for each datapoint type tab
            for i, dtype in enumerate(datapoint_types):
                html_parts.append(f'<div id="tab-content-{query_id}-{i}" class="tab-content">')

                type_df = query_df[query_df["datapoint_type"] == dtype]
                type_count = len(type_df)

                # Create sub-tabs for Top and Sample (if percentile_bounds provided)
                if percentile_bounds is not None:
                    # Sub-tab buttons
                    html_parts.append('<div class="tab-container" style="margin-top: 0;">')
                    html_parts.append('<div class="tab-buttons">')
                    html_parts.append(
                        f'<button id="subtab-button-{query_id}-{i}-top" class="tab-button" onclick="switchTab(\'{query_id}-{i}\', \'top\')">Top {n_train}</button>'
                    )
                    html_parts.append(
                        f'<button id="subtab-button-{query_id}-{i}-sample" class="tab-button" onclick="switchTab(\'{query_id}-{i}\', \'sample\')">Sample ({percentile_bounds[0]:.0f}-{percentile_bounds[1]:.0f}%ile)</button>'
                    )
                    html_parts.append("</div>")

                    # Top documents sub-tab
                    html_parts.append(f'<div id="tab-content-{query_id}-{i}-top" class="tab-content">')

                    if plot_bottom_instead:
                        type_sorted = type_df.nsmallest(min(n_train, type_count), "influence_score")
                        direction = "lowest"
                    else:
                        type_sorted = type_df.nlargest(min(n_train, type_count), "influence_score")
                        direction = "highest"

                    html_parts.append(
                        f'<div class="tab-info">Showing top {min(n_train, type_count)} {direction} scoring documents of type "{dtype}" (out of {type_count} total)</div>'
                    )

                    html_parts.extend(
                        create_content_for_group(
                            type_sorted,
                            query_id,
                            train_id_to_idx,
                            train_dataset_df,
                            q_item,
                            tokenizer,
                            escape_html,
                            get_color_rgb,
                            check_shared_entity,
                            f"type-{i}-top",
                        )
                    )
                    html_parts.append("</div>")  # Close top sub-tab

                    # Sample sub-tab
                    html_parts.append(f'<div id="tab-content-{query_id}-{i}-sample" class="tab-content">')

                    sample_df = get_percentile_sample(type_df, percentile_bounds, n_train, plot_bottom_instead)
                    sample_count = len(sample_df)

                    # Calculate how many documents are in the percentile range
                    scores = type_df["influence_score"].values
                    lower_threshold = np.percentile(scores, percentile_bounds[0])
                    upper_threshold = np.percentile(scores, percentile_bounds[1])
                    in_range_count = len(
                        type_df[
                            (type_df["influence_score"] >= lower_threshold)
                            & (type_df["influence_score"] <= upper_threshold)
                        ]
                    )

                    html_parts.append(
                        f'<div class="tab-info">Random sample of {sample_count} documents from {percentile_bounds[0]:.0f}-{percentile_bounds[1]:.0f} percentile range of type "{dtype}" ({in_range_count} total in range, {type_count} total)</div>'
                    )

                    html_parts.extend(
                        create_content_for_group(
                            sample_df,
                            query_id,
                            train_id_to_idx,
                            train_dataset_df,
                            q_item,
                            tokenizer,
                            escape_html,
                            get_color_rgb,
                            check_shared_entity,
                            f"type-{i}-sample",
                        )
                    )
                    html_parts.append("</div>")  # Close sample sub-tab
                    html_parts.append("</div>")  # Close sub-tab container

                else:
                    # Original single tab content when no percentile_bounds
                    if plot_bottom_instead:
                        type_sorted = type_df.nsmallest(min(n_train, type_count), "influence_score")
                        direction = "lowest"
                    else:
                        type_sorted = type_df.nlargest(min(n_train, type_count), "influence_score")
                        direction = "highest"

                    html_parts.append(
                        f'<div class="tab-info">Showing top {min(n_train, type_count)} {direction} scoring documents of type "{dtype}" (out of {type_count} total)</div>'
                    )

                    html_parts.extend(
                        create_content_for_group(
                            type_sorted,
                            query_id,
                            train_id_to_idx,
                            train_dataset_df,
                            q_item,
                            tokenizer,
                            escape_html,
                            get_color_rgb,
                            check_shared_entity,
                            f"type-{i}",
                        )
                    )

                html_parts.append("</div>")  # Close main type tab

            html_parts.append("</div>")  # Close tab-container

        else:
            # Original behavior: show overall top documents
            if plot_bottom_instead:
                query_df_sorted = query_df.nsmallest(n_train, "influence_score")
            else:
                query_df_sorted = query_df.nlargest(n_train, "influence_score")

            html_parts.extend(
                create_content_for_group(
                    query_df_sorted,
                    query_id,
                    train_id_to_idx,
                    train_dataset_df,
                    q_item,
                    tokenizer,
                    escape_html,
                    get_color_rgb,
                    check_shared_entity,
                    "main",
                )
            )

        html_parts.append("</div>")  # Close container
        html_parts.append("</body></html>")

        html_docs[f"query_{query_id}"] = "\n".join(html_parts)

    return html_docs


def create_content_for_group(
    sorted_df,
    query_id,
    train_id_to_idx,
    train_dataset_df,
    q_item,
    tokenizer,
    escape_html,
    get_color_rgb,
    check_shared_entity,
    tab_suffix,
):
    """Helper function to create HTML content for a group of documents"""
    html_parts = []

    # Calculate sum of influence scores and valid documents
    doc_sums = []
    valid_train_ids = []
    for _, row in sorted_df.iterrows():
        train_id = row["train_id"]
        if str(train_id) in train_id_to_idx:  # Only include if train_id exists in mapping
            tok_scores = row["per_token_influence_score"]
            doc_sum = np.sum(tok_scores)
            doc_sums.append(doc_sum)
            valid_train_ids.append(train_id)

    # Create bar chart
    html_parts.append('<div class="bar-chart-container">')
    html_parts.append(
        f'<div class="bar-chart-title">Sum of Influence Scores by Document Rank (Click to Navigate) - Showing {len(valid_train_ids)} of {len(sorted_df)} documents</div>'
    )
    html_parts.append('<div class="bar-chart">')
    html_parts.append('<div class="y-axis-label">Sum Score</div>')

    # Normalize bar heights
    if doc_sums:
        max_abs_sum = max(abs(s) for s in doc_sums)
        if max_abs_sum > 0:
            normalized_heights = [abs(s) / max_abs_sum * 180 for s in doc_sums]  # 180px max height
        else:
            normalized_heights = [0] * len(doc_sums)
    else:
        normalized_heights = []

    for rank, (sum_score, height, train_id) in enumerate(zip(doc_sums, normalized_heights, valid_train_ids)):
        bar_class = "bar-positive" if sum_score >= 0 else "bar-negative"
        doc_id = f"doc-{query_id}-{train_id}-{tab_suffix}"

        html_parts.append(f'<div class="bar" onclick="scrollToDocument(\'{doc_id}\')">')
        html_parts.append(f'<div class="bar-fill {bar_class}" style="height: {height}px;"></div>')
        html_parts.append(f'<div class="bar-label">{rank + 1}</div>')
        html_parts.append(
            f'<div class="bar-tooltip">Rank {rank + 1}<br>Sum: {sum_score:+.3f}<br>Click to navigate</div>'
        )
        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append('<div class="x-axis-label">Document Rank</div>')
    html_parts.append("</div>")

    # Summary table
    html_parts.append('<table class="summary-table">')
    html_parts.append("<thead><tr>")
    html_parts.append(
        "<th>Rank</th><th>IF-Score</th><th>Sum</th><th>Type</th><th>Parent?</th><th>Shared Entity?</th><th>Preview</th>"
    )
    html_parts.append("</tr></thead><tbody>")

    actual_rank = 0
    for original_rank, (_, row) in enumerate(sorted_df.iterrows()):
        train_id = row["train_id"]
        if str(train_id) not in train_id_to_idx:
            print(f"Warning: train_id {train_id} not found in train_id_to_idx mapping. Skipping.")
            continue
        t_idx = train_id_to_idx[str(train_id)]

        t_item = train_dataset_df.iloc[t_idx]

        # Check if this is the parent fact (use datapoint_type from influence scores)
        is_parent = row.get("datapoint_type", "") == "parent_fact"

        # Check if they share the same entity
        has_shared_entity = check_shared_entity(q_item, t_item)

        if_score = row["influence_score"]
        sum_score = doc_sums[original_rank]  # Use original rank for doc_sums indexing
        datapoint_type = row.get("datapoint_type", t_item["type"])

        score_class = "positive-score" if if_score >= 0 else "negative-score"
        sum_class = "positive-score" if sum_score >= 0 else "negative-score"
        parent_html = (
            '<span class="parent-tag">PARENT</span>' if is_parent else '<span class="not-parent-tag">NOT-PARENT</span>'
        )
        entity_html = (
            '<span class="shared-entity-tag">SHARED</span>'
            if has_shared_entity
            else '<span class="no-shared-entity-tag">DIFFERENT</span>'
        )

        preview = (t_item.get("prompt") or "")[:35].replace("\n", " ")

        html_parts.append("<tr>")
        html_parts.append(f"<td>{actual_rank + 1}</td>")  # Use actual_rank for display
        html_parts.append(f'<td class="{score_class}">{if_score:+.3f}</td>')
        html_parts.append(f'<td class="{sum_class}">{sum_score:+.3f}</td>')
        html_parts.append(f"<td>{escape_html(datapoint_type)}</td>")
        html_parts.append(f"<td>{parent_html}</td>")
        html_parts.append(f"<td>{entity_html}</td>")
        html_parts.append(f"<td>{escape_html(preview)}</td>")
        html_parts.append("</tr>")
        actual_rank += 1

    html_parts.append("</tbody></table>")

    # Per-token influence scores
    html_parts.append('<div class="token-section">')
    html_parts.append('<div class="token-header">Per-token IF scores:</div>')

    # Add view toggle button
    html_parts.append('<div class="view-toggle-container">')
    html_parts.append('<div class="view-toggle-label">Display Mode:</div>')
    html_parts.append(
        f'<button id="view-toggle-{query_id}-{tab_suffix}" class="toggle-button" onclick="toggleViewMode(\'{query_id}\', \'{tab_suffix}\')" title="Switch between token blocks and continuous text view">Switch to Text View</button>'
    )
    html_parts.append("</div>")

    actual_rank = 0
    for original_rank, (_, row) in enumerate(sorted_df.iterrows()):
        train_id = row["train_id"]
        if str(train_id) not in train_id_to_idx:
            print(f"Warning: train_id {train_id} not found in train_id_to_idx mapping. Skipping.")
            continue
        t_idx = train_id_to_idx[str(train_id)]

        t_item = train_dataset_df.iloc[t_idx]
        input_ids = t_item["input_ids"]

        tok_scores = row["per_token_influence_score"]
        tok_strings = tokenizer.batch_decode(input_ids)

        # Calculate statistics for this document's influence scores
        scores_array = np.array(tok_scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        sum_score = np.sum(scores_array)
        if std_score < 1e-7:  # Avoid division by zero
            std_score = 1e-7

        # Create unique ID for this document's token display and navigation
        doc_display_id = f"doc_{query_id}_{train_id}_{tab_suffix}"
        doc_nav_id = f"doc-{query_id}-{train_id}-{tab_suffix}"

        # Header with type and toggle button (with navigation anchor)
        html_parts.append(f'<div id="{doc_nav_id}" class="doc-header">')
        html_parts.append("<div>")
        html_parts.append(f"<span>--- Train doc #{train_id} (rank {actual_rank + 1}) ---</span>")  # Use actual_rank
        html_parts.append(f' <span class="doc-type">{escape_html(row.get("datapoint_type", t_item["type"]))}</span>')
        html_parts.append("</div>")
        html_parts.append(
            f'<button id="toggle-{doc_display_id}" class="toggle-button" onclick="toggleScoreView(\'{doc_display_id}\')">Show All Scores</button>'
        )
        html_parts.append("</div>")

        html_parts.append(
            f'<div class="stats-info">Mean: {mean_score:+.4f}, Std Dev: {std_score:.4f}, Sum: {sum_score:+.4f}</div>'
        )
        actual_rank += 1

        # Token display with colors
        html_parts.append(f'<div id="tokens-{doc_display_id}" class="token-display">')

        for tok, sc in zip(tok_strings, tok_scores):
            score_float = float(sc)
            std_devs = (score_float - mean_score) / std_score

            # Handle newlines in tokens
            if "\n" in tok:
                tok_parts = tok.split("\n")
                for i, part in enumerate(tok_parts):
                    if i > 0:
                        html_parts.append("<br>")
                    if part:  # Only add non-empty parts
                        bg_color, text_color = get_color_rgb(score_float, std_devs)

                        html_parts.append('<span class="token-with-score">')
                        html_parts.append(
                            f'<span class="token-cell" style="background-color: {bg_color}; color: {text_color};" '
                            f'title="Score: {score_float:+.4f} ({std_devs:+.2f}σ)">'
                            f"{escape_html(part)}"
                        )

                        # Add std dev indicator for significant deviations (only shown in default view)
                        if abs(std_devs) >= 2.0:
                            html_parts.append(f'<span class="std-dev-indicator">{std_devs:+.0f}σ</span>')

                        html_parts.append("</span>")

                        # Add score label (only shown in detailed view)
                        html_parts.append(f'<span class="score-label">{score_float:+.3f}<br>{std_devs:+.1f}σ</span>')
                        html_parts.append("</span>")
            else:
                bg_color, text_color = get_color_rgb(score_float, std_devs)

                html_parts.append('<span class="token-with-score">')
                html_parts.append(
                    f'<span class="token-cell" style="background-color: {bg_color}; color: {text_color};" '
                    f'title="Score: {score_float:+.4f} ({std_devs:+.2f}σ)">'
                    f"{escape_html(tok)}"
                )

                # Add std dev indicator for significant deviations (only shown in default view)
                if abs(std_devs) >= 2.0:
                    html_parts.append(f'<span class="std-dev-indicator">{std_devs:+.0f}σ</span>')

                html_parts.append("</span>")

                # Add score label (only shown in detailed view)
                html_parts.append(f'<span class="score-label">{score_float:+.3f}<br>{std_devs:+.1f}σ</span>')
                html_parts.append("</span>")

        html_parts.append("</div>")

        # Score display
        html_parts.append('<div class="score-display">')
        html_parts.append(
            "<small>Default view: Hover for scores, ≥2σ tokens marked. Toggle button shows/hides all scores.</small>"
        )
        html_parts.append("</div>")

    html_parts.append("</div>")  # Close token-section
    return html_parts
