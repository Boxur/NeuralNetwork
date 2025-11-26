#pragma once
#include <string>
#include <stdio.h>
#include <thread>
#include <mutex>
#include <chrono>

inline class logger
{
public:
	enum LogLevel
	{
		info = 0x01,
		warning = 0x02,
		error = 0x04,
		debug = 0x08,
		memory = 0x10
	};

private:
	short _level = info | warning | error | debug | memory;
	std::mutex _mutex;

public:
	void operator()(LogLevel level, std::string text);

	void operator()(std::string text);

	void Level(short level);

private:
	void LOG(std::string text, LogLevel level);

	std::string logLevelToText(LogLevel);
} Log;
