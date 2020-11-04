//
// Created by KlausP on 20.10.2020.
//

#ifndef OCTREECUDA_EVENTWATCHER_H
#define OCTREECUDA_EVENTWATCHER_H

// Standard library
#include <fstream>

// Local dependencies
#include "spdlog/spdlog.h"


using namespace std::chrono;


constexpr double GB = 1000000000.0;


class EventWatcher {

public:
    static EventWatcher& getInstance()
    {
        static EventWatcher instance;
        return instance;
    }

    ~EventWatcher() {
        std::string data;
        std::string labels;

        for(auto i = 0; i < itsMemoryFootprints.size(); ++i) {
            data += std::to_string(itsMemoryFootprints[i]);
            labels += ("\"" + itsEventLabels[i] + "\"");
            if(i < itsMemoryFootprints.size() - 1) {
                data += ",";
                labels += ",";
            }
        }

        std::ofstream htmlData;
        htmlData.open (std::string(itsFilename), std::ios::out);
        htmlData << itsHtmlPart1;
        htmlData << "{labels: [" << labels << "], datasets: [{label: \"Memory footprint\", backgroundColor: "
                                              "\"rgb(255, 99, 132)\", borderColor: \"rgb(255, 99, 132)\",";
        htmlData << "data: [" << data << "]}]},";
        htmlData << itsHtmlPart2;
        htmlData.close();
    }

    void configureMemoryReport(const std::string &filename) {
        itsFilename = filename;
    }

    void reservedMemoryEvent(uint64_t memoryFootprint, const std::string& name) {
        memoryConsumption += (memoryFootprint / GB);
        recordEvent(name);
    }

    void freedMemoryEvent(uint64_t memoryFootprint, const std::string& name) {
        memoryConsumption -= (memoryFootprint / GB);
        recordEvent("Deleted " + name);
    }

private:

    EventWatcher() {
        memoryConsumption = 0;
        itsFilename = "memory_report.html";
    }

    void recordEvent(const std::string& name) {
        itsEventLabels.push_back(name);
        itsMemoryFootprints.push_back(memoryConsumption);
    }

public:
    EventWatcher(EventWatcher const&)  = delete;
    void operator=(EventWatcher const&) = delete;

private:
    double memoryConsumption;
    std::vector<double> itsMemoryFootprints;
    std::vector<std::string> itsEventLabels;
    time_point<steady_clock> itsStart;
    std::string itsFilename;

    const std::string itsHtmlPart1 =
            "<html>\n"
            "    <head>\n"
            "        <script src=\"https://cdn.jsdelivr.net/npm/chart.js@2.8.0\"></script>\n"
            "        <style>\n"
            "            html, body {\n"
            "            margin: 0;\n"
            "            height: 100%;\n"
            "            }\n"
            "            canvas {\n"
            "            width: 100%;\n"
            "            height: 100%;\n"
            "            display: block;\n"
            "            }\n"
            "        </style>\n"
            "    </head>\n"
            "    <body>\n"
            "        <canvas id=\"myChart\"></canvas>\n"
            "        <script>\n"
            "            var ctx = document.getElementById('myChart').getContext('2d');\n"
            "            var chart = new Chart(ctx, {\n"
            "                // The type of chart we want to create\n"
            "                type: 'line',\n"
            "\n"
            "                // The data for our dataset\n"
            "                data:";

    const std::string itsHtmlPart2 =
            "                // Configuration options go here\n"
            "                options: {\n"
            "                    scales: {\n"
            "                        yAxes: [{\n"
            "                            ticks: {\n"
            "                                beginAtZero: true\n"
            "                            },\n"
            "                            scaleLabel: {\n"
            "                               display: true,"
            "                               labelString: 'Memory usage [GB]'"
            "                            }"
            "                        }]\n"
            "                    }\n"
            "                },\n"
            "\n"
            "            });\n"
            "        </script>\n"
            "    </body>\n"
            "</html>";
};

#endif //OCTREECUDA_EVENTWATCHER_H