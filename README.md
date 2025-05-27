
# ER 2025 Data Analysis Script

This repository contains a Python script developed to automate the data processing and analysis workflows used in a manuscript submitted to ER 2025. The script is designed to extract data from the OntoUML/UFO Catalog, apply selection criteria, and calculate statistics used in the quantitative analyses described in the paper. It ensures the reproducibility of results, generates structured outputs (e.g., CSV files), and produces the visualizations and graphs included in the study.

## Repository Purpose

This repository serves two main purposes:
1. **Reproducibility**: Allow users to replicate the analyses described in the ER 2025 paper.
2. **Data Accessibility**: Provide access to the data used in the paper.

## Getting Started

To replicate the analysis, follow these steps:

### 1. Clone This Repository
Open a terminal and execute the following command:
```bash
git clone https://github.com/pedropaulofb/er2025data.git
```

### 2. Clone the OntoUML/UFO Catalog
You can [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the OntoUML/UFO Catalog in one of two ways:

#### a) Use the Latest Catalog Version
To perform an analysis with the most up-to-date information:
```bash
git clone https://github.com/OntoUML/ontouml-models.git
```

#### b) Use the Exact Version for Reproducibility
To reproduce the exact data used in the paper:
1. Download the repository at the specific commit using this [ZIP snapshot link](https://github.com/OntoUML/ontouml-models/archive/0b62aa42171fdcfaa0d0533b516012eda66f3ad6.zip).
2. Extract the ZIP file.
3. Rename the extracted folder to `ontouml-models`.

### 3. Set Up the Folder Structure
Ensure both repositories are placed in the same root directory. For example:
```
my_folder/
├── er2025data/
├── ontouml-models/
```

### 4. Install Dependencies
Navigate to the `er2025data` directory and install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 5. Run the Script
To execute the script and generate the analysis:
```bash
python main.py
```

## Output

After the execution, the software will create an output folder (e.g., `er2025data/output`). Inside this folder, the results will be organized as follows:

- **Calculated Statistics**: Available in `output/02_datasets_statistics`.
- **Graphs and Visualizations**: Available in `output/03_visualizations`.

The data used in the paper is available in the output folder of this repository, ensuring validation by the community.



## Developer

This script was developed by:

<table>
  <tr>
    <td><strong>Pedro Paulo F. Barcelos</strong></td>
    <td>
      <a href="https://orcid.org/0000-0003-2736-7817"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" alt="ORCID" width="20"/></a>
      <a href="https://github.com/pedropaulofb"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" width="20"/></a>
      <a href="https://www.linkedin.com/in/pedro-paulo-favato-barcelos/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="20"/></a>
    </td>
  </tr>
</table>