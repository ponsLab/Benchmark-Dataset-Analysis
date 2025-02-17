# ExoGait-Benchmark
This repository provides code to explore our benchmark dataset for lower-limb exoskeleton control (cite) and example of performance analysis across various ambulation modes.


![GIF showing 4 locomotion modes using a lower-limb exoskeleton in SRALAB](assets/exo_locomotion.gif)

## Table of Contents

* [About the Dataset](#about-the-dataset)
* [Data Structure](#data-structure)
* [Usage Instructions](#usage-instructions)
    * [Dependencies](#dependencies)
    * [Running the Code](#running-the-code)
    * [Example Scripts](#example-scripts)
* [Performance Analysis](#performance-analysis)
    * [Metrics](#metrics)
    * [Visualization](#visualization)
* [Contributing](#contributing)
* [License](#license)
* [Citation](#citation)
* [Contact](#contact)

## About the Dataset

    Purpose: Briefly explain the purpose of the dataset (e.g., to benchmark different control algorithms for lower-limb exoskeletons).
    Collection Methodology: Describe how the data was collected (e.g., experimental setup, participants, ambulation modes, sensors used).
    Data Size and Format: Specify the size of the dataset, the format of the data files (e.g., CSV, MAT), and any relevant data organization details.

## Data Structure
The associated dataset is available as supplemental material and can be accessed through the provided link (\url{DOI: 10.5061/dryad.70rxwdc6j}). Each participant's data is stored in a separate file, which contains tables detailing the iterations and activities performed.
Each participant performed activities described in Sec. \ref{sec:experimentalSetup} while Tab. \ref{tab:structureData} displays the features (in columns) for each file. 
Participants started the session with overground walking, followed by stairs and ramps in a randomized order. During each locomotion activity, the TR and SM conditions were tested, with their order randomized as well. Each file contains in chronological order all there activities performed by a single participant while the rest time between consecutive activities has been removed. The data was recorded and stored at 333~Hz. The columns in the header of each file contain sensor information, and the rows indicate time in seconds. Some of the information collected from the exoskeleton are: the joint position ($\bm{q}$), joint velocity ($\dot{\bm{q}}$), joint torque ($\bm{\tau}$), interaction forces ($\bm{\tau}_{int}$), stance interpolation factor ($\alpha$). The gait phase for each data point was determined using linear interpolation between heel strikes, covering a range from 0\% to 100\% of the gait cycle, with heel strikes identified from the force sensor readings. EMG information is available only for the five subjects collected at Hospital Los Madroños, stored in separate files with corresponding subject identifiers. 

![Features collected with exoskeleton locomotion (TR&SM)](assets/Tab-3.png)

## Usage Instructions

### Dependencies: see requirements.txt 
### Running the Code: 
- copy the dataset [10.5061/dryad.70rxwdc6j] in the /data folder
- create an environment thanks to requirements.txt file
- Explore the dataset for a specific user thanks to `Explore_entire_dataset.ipynb`
- Explore the entire dataset with `Explore_entire_dataset.ipynb`

- `assests/helpers.py` define different classes/function helpful for data exploration
## Performance Analysis

Provided functions in the different notebooks are focusing on ...

## Contributing

    Contribution Guidelines: Outline how others can contribute to the repository (e.g., bug reports, feature requests, code contributions).
    Coding Style: If applicable, specify any coding style guidelines to be followed.

## License

    License Type: Clearly state the license under which the code and dataset are released (e.g., MIT License, GNU GPL).

## Citation

    BibTeX Entry: Provide a BibTeX entry for citing the paper associated with the dataset and code.

## Contact

    Contact Information: Provide contact information for inquiries or support (e.g., email address, GitHub username).