#include "mnist_data.hpp"
#include "network.hpp"
#include "log.hpp"
#include <memory>

int main()
{
	//this is a comment
	std::shared_ptr<NetworkData> data = std::make_shared<MnistData>();
	Network network(data, 0.000001f);
	network.LoadNetwork("network.nn");
	Log.Level(Log.error | Log.info);
	network.Train(10);
	network.Test();
	network.SaveNetwork("network.nn");
}

