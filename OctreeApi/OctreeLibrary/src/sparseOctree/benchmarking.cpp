//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>


void SparseOctree::exportTimeMeasurements(const string &filePath) {
    string headerLine, timeLine;
    ofstream timingCsv;
    timingCsv.open (filePath, ios::out);
    for (auto const& timeEntry : itsTimeMeasurement) {
        headerLine += (timeEntry.first + ",");
        timeLine += (to_string(timeEntry.second) + ",");
    }
    // Remove last colons
    headerLine = headerLine.substr(0, headerLine.size()-1);
    timeLine = timeLine.substr(0, timeLine.size()-1);
    timingCsv << headerLine << std::endl << timeLine;
    timingCsv.close();
    spdlog::info("Exported time measurements to {}", filePath);
}