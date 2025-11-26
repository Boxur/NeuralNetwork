#include "log.h"

void logger::operator()(LogLevel level, std::string text)
{
	if ((_level & level) != level||text=="")
		return;
	std::thread thread(&logger::LOG,this, text,level);
	thread.detach();
}

void logger::operator()(std::string text)
{
	this->operator()(this->info,text);
}

void logger::Level(short level)
{
	_level = level;
} 

void logger::LOG(std::string text,LogLevel level)
{
	
	std::chrono::system_clock::time_point td = std::chrono::system_clock::now();
	auto td_days = std::chrono::time_point_cast<std::chrono::duration<int, std::ratio<86400>>>(td);
	auto now = td - td_days;
	auto h = std::chrono::duration_cast<std::chrono::hours>(now);
	now -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(now);
	now -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(now);
	now -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now);

	std::lock_guard<std::mutex> guard(_mutex);

	printf
	(	
		"(%02d:%02d:%02d.%03d) [%s]\b\t %s\n",
		(int)h.count(),
		(int)m.count(),
		(int)s.count(),
		(int)ms.count(),
		logLevelToText(level).c_str(),
		text.c_str()
	);

}

std::string logger::logLevelToText(LogLevel level)
{
	std::string ans;
	switch (level)
	{
	case logger::info:
		return "\x1B[34mInfo\033[0m";
		break;
	case logger::warning:
		return "\x1B[33mWarning\033[0m";
		break;
	case logger::error:
		return "\x1B[31mError\033[0m";
		break;
	case logger::debug:
		return "\x1B[32mDebug\033[0m";
		break;
	case logger::memory:
		return "\x1B[35mMemory\033[0m";
		break;
	default:
		return "Unknown Type";
		break;
	}
}
