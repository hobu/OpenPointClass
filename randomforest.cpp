#include "randomforest.hpp"


#include <pdal/util/Extractor.hpp>
#include <pdal/util/Inserter.hpp>

#include <filesystem>
namespace fs = std::filesystem;

namespace rf {

bool loadVectors(const std::string &filename,
                 int& numScales,
                 int& numTrees,
                 int& treeDepth,
                 double& radius,
                 int& maxSamples,
                 std::vector<int>& classes,
                 double& startResolution,
                 std::vector<int>& gt,
                 std::vector<float>& ft)
{
    std::ifstream ifs(filename.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!ifs.good())
        throw std::runtime_error ("Unable to open existing vector file!");

    ifs.seekg(0);

    ifs.read(reinterpret_cast<char *>(&numScales), sizeof (numScales));
    std::cout << "numScales: " << numScales << std::endl;

    ifs.read(reinterpret_cast<char *>(&numTrees), sizeof (numTrees));
    std::cout << "numTrees: " << numTrees << std::endl;

    ifs.read(reinterpret_cast<char *>(&treeDepth), sizeof (treeDepth));
    std::cout << "treeDepth: " << treeDepth << std::endl;

    ifs.read(reinterpret_cast<char *>(&radius), sizeof (radius));
    std::cout << "radius: " << radius << std::endl;

    ifs.read(reinterpret_cast<char *>(&maxSamples), sizeof (maxSamples));
    std::cout << "maxSamples: " << maxSamples << std::endl;

    if (classes.size())
        throw std::runtime_error("Classification values already populated!");

    int size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof (size));
    std::cout << "classification size: " << size << std::endl;

    for (size_t i = 0; i < size; ++i)
    {
        int v;
        ifs.read(reinterpret_cast<char *>(&v), sizeof (v));
        std::cout << "classification value: " << v << std::endl;
        classes.push_back(v);
    }


    // gt size now
    ifs.read(reinterpret_cast<char *>(&size), sizeof (size));
    for (size_t i = 0; i < size; ++i)
    {
        int v;
        ifs.read(reinterpret_cast<char *>(&v), sizeof (v));
        gt.push_back(v);
    }

    // ft size now
    ifs.read(reinterpret_cast<char *>(&size), sizeof (size));
    for (size_t i = 0; i < size; ++i)
    {
        float v;
        ifs.read(reinterpret_cast<char *>(&v), sizeof (v));
        ft.push_back(v);
    }

    // startResolution
    ifs.read(reinterpret_cast<char*>(&startResolution), sizeof(double));
    std::cout << "startResolution: " << startResolution << std::endl;
    return true;

}


bool isTrainingVectors(const std::string& filename)
{
    fs::path p(filename);
    fs::path basename = p.stem();

    if (basename.string().find("opc-features") == std::string::npos)
        return false;

    if (!fs::exists(p))
        return false;

    return true;
}

RandomForest *train(const std::vector<std::string> &filenames,
    double *startResolution,
    const int numScales,
    const int numTrees,
    const int treeDepth,
    const double radius,
    const int maxSamples,
    const std::vector<int> &classes,
    bool interrupt) {

    ForestParams params;
    params.n_trees = numTrees;
    params.max_depth = treeDepth;
    auto *rtrees = new RandomForest(params);
    const AxisAlignedRandomSplitGenerator generator;

    std::vector<float> ft;
    std::vector<int> gt;

    bool bExistingVectors(false);
    for (auto& f: filenames)
    {
        std::vector<float> l_ft;
        std::vector<int> l_gt;
        int l_numScales;
        int l_numTrees;
        int l_treeDepth;
        double l_radius;
        int l_maxSamples;
        double l_startResolution;
        std::vector<int> l_classes;

        if (isTrainingVectors(f))
        {
            std::cout << "loading existing vectors for " << f << std::endl;
            // load training data
            bool loaded = loadVectors(f,
                                        l_numScales,
                                        l_numTrees,
                                        l_treeDepth,
                                        l_radius,
                                        l_maxSamples,
                                        l_classes,
                                        l_startResolution,
                                        l_gt,
                                        l_ft);
            std::cout << "loaded existing vectors " << loaded << std::endl;
            std::cout << "this file's gt.size() " << l_gt.size() << std::endl;
            std::cout << "this file's ft.size() " << l_ft.size() << std::endl;
            std::cout << "global gt.size() " << gt.size() << std::endl;
            std::cout << "global ft.size() " << ft.size() << std::endl;


            std::copy (l_ft.begin(), l_ft.end(), std::back_inserter(ft));
            std::copy (l_gt.begin(), l_gt.end(), std::back_inserter(gt));
            std::cout << "appended global gt.size() " << gt.size() << std::endl;
            std::cout << "appended global ft.size() " << ft.size() << std::endl;
            bExistingVectors = true;
            std::cout << "l_startResolution " << l_startResolution << std::endl;
            *startResolution = l_startResolution;
        }
    }

    if (!bExistingVectors)
    {
        getTrainingData(filenames, startResolution, numScales, radius, maxSamples, classes,
            [&ft, &gt](const std::vector<Feature *> &features, size_t idx, int g) {
                for (std::size_t f = 0; f < features.size(); f++) {
                    ft.push_back(features[f]->getValue(idx));
                }
                gt.push_back(g);
            },
            [](size_t numFeatures, int numClasses) {});
    }
    std::cout << "Using " << gt.size() << " inliers" << std::endl;

    if (interrupt && !bExistingVectors)
    {
        saveVectors("opc-vectors.bin",
                    numScales,
                    numTrees,
                    treeDepth,
                    radius,
                    maxSamples,
                    classes,
                    *startResolution,
                    gt,
                    ft);
        return nullptr;
    }

    const LabelDataView label_vector(gt.data(), gt.size(), 1);
    const FeatureDataView feature_vector(ft.data(), gt.size(), ft.size() / gt.size());

    std::cout << "Training..." << std::endl;
    rtrees->train(feature_vector, label_vector, LabelDataView(), generator, 0, false, false);

    rtrees->params.resolution = *startResolution;
    rtrees->params.radius = radius;
    rtrees->params.numScales = numScales;

    return rtrees;
}

void saveVectors(const std::string& filename,
                    int numScales,
                    int numTrees,
                    int treeDepth,
                    double radius,
                    int maxSamples,
                    const std::vector<int> &classes,
                    double startResolution,
                    const std::vector<int>& gt,
                    const std::vector<float>& ft)
{
    // Take the first filename, grab its basename, and add 'opc-features' to it
    fs::path p(filename);
    fs::path basename = p.stem();
    std::string output = "opc-features.bin";
//     std::string output = basename.string() + "-opc-features" + ".bin";

    std::ofstream ofs(output.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    if (!ofs.good())
        throw std::runtime_error ("Unable to open vector file for writing!");


    std::cout << "numScales: " << numScales << std::endl;

    std::cout << "numTrees: " << numTrees << std::endl;

    std::cout << "treeDepth: " << treeDepth << std::endl;

    std::cout << "radius: " << radius << std::endl;

    std::cout << "maxSamples: " << maxSamples << std::endl;

    ofs.write((char*) &numScales, sizeof(int));

    // Write numTrees
    ofs.write((char*) &numTrees, sizeof(int));

    // Write treeDepth
    ofs.write((char*) &treeDepth, sizeof(int));

    // Write radius
    ofs.write((char*) &radius, sizeof(double));

    // Write maxSamples
    ofs.write((char*) &maxSamples, sizeof(int));

    size_t size(classes.size());
    ofs.write((char*) &size, sizeof(int));
    for (size_t i = 0; i < size; ++i)
        ofs.write((char*) &classes[i], sizeof(int));

    // How many gt
    size = gt.size();
    ofs.write((char*) &size, sizeof(int));
    for (size_t i = 0; i < size; ++i)
        ofs.write((char*) &gt[i], sizeof(int));

    // How many ft
    size = ft.size();
    ofs.write((char*) &size, sizeof(int));
    for (size_t i = 0; i < size; ++i)
        ofs.write((char*) &ft[i], sizeof(float));


    ofs.write((char*) &startResolution, sizeof(double));

    ofs.flush();
    ofs.close();


}

void saveForest(RandomForest *rtrees, const std::string &modelFilename) {
    std::ofstream ofs(modelFilename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    rtrees->write(ofs);

    std::cout << "Saved " << modelFilename << std::endl;
}

RandomForest *loadForest(const std::string &modelFilename) {
    const auto rtrees = new RandomForest();

    std::cout << "Loading " << modelFilename << std::endl;
    std::ifstream ifs(modelFilename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open " + modelFilename);
    rtrees->read(ifs);

    return rtrees;
}

void classify(PointSet &pointSet,
    RandomForest *rtrees,
    const std::vector<Feature *> &features,
    const std::vector<Label> &labels,
    const Regularization regularization,
    const double regRadius,
    const bool useColors,
    const bool unclassifiedOnly,
    const bool evaluate,
    const std::vector<int> &skip,
    const std::string &statsFile) {
    classifyData<float>(pointSet,
        [&rtrees](const float *ft, float *probs) {
            rtrees->evaluate(ft, probs);
        },
        features, labels, regularization, regRadius, useColors, unclassifiedOnly, evaluate, skip, statsFile);
}

}
