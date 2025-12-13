#pragma once
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

class Layer
{
public:
	enum class DeltaMode
	{
		Diffrence,
		Multiplication
	};

	enum class Type
	{
		Hidden,
		Classification,
		Regression
	};
	

private:
	int inputCount_;
	int outputCount_;
	std::vector<double> outputs_;
	std::vector<double> biases_;
	std::vector<double> weights_;
	std::vector<double> deltas_;

	Type type_;

public:
	Layer(int inputs, int outputs, Type type);

	~Layer();

	void InitWeights();

	double ActivationFunction(double x) const;

	double ActivationFunctionDerivative(double x) const;

	std::vector<double> Compute(const std::vector<double> &inputs);

	void Delta(std::vector<double>& output, DeltaMode mode);

	void Descent(const std::vector<double>& inputs, double learningRate);

	double Random();

	void SaveLayer(std::fstream &file);

	void LoadLayer(std::fstream &file);

	std::vector<double>& GetOutputs();
};
