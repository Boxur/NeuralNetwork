#pragma once
#include "layer.hpp"
#include "network_data.hpp"
#include "log.hpp"

#include <assert.h>
#include <fstream>
#include <vector>
#include <memory>

class Network
{
private:
	double learningRate_;
	int inputCount_;
	int outputCount_;
	int layerCount_;
	int biggestLayer_;
	std::vector<std::unique_ptr<Layer>> layers_;
	std::shared_ptr<NetworkData> data_;

	double precission_;

public:
	Network(std::shared_ptr<NetworkData> data, double learningRate);

	~Network();

	void Train(int epochs);

	void Train();

	std::vector<double> Compute(const std::vector<double>& inputs);

	void SaveNetwork(const std::string& path);

	bool LoadNetwork(const std::string& path);

	void Test();

private:
	void Train_(std::vector<double>& inputs,std::vector<double>& outputs);

	static double CalculateError_(const std::vector<double>& expectedOutputs, const std::vector<double>& calculatedOutputs, int outputCount);

	double TestNetwork_();

	void Backpropagation_(std::vector<double>& inputs, std::vector<double>& outputs);
};
