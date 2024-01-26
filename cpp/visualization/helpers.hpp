#include <string>
#include <chrono>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#ifdef _MSC_VER
#include <direct.h>
#endif

int makeDir(const std::string &dir) {
#ifdef _MSC_VER
    return _mkdir(dir.c_str());
#else
    mode_t mode = 0755;
    return mkdir(dir.c_str(), mode);
#endif
}

bool folderExists(const std::string &folder) {
#ifdef _MSC_VER
    struct _stat info;
    if (_stat(folder.c_str(), &info) != 0) return false;
    return (info.st_mode & _S_IFDIR) != 0;
#else
    struct stat info;
    if (stat(folder.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
#endif
}

bool createFolders(const std::string &folder) {
    int ret = makeDir(folder);
    if (ret == 0) return true;

    switch (errno) {
        case ENOENT: {
            size_t pos = folder.find_last_of('/');
            if (pos == std::string::npos)
#ifdef _MSC_VER
                pos = folder.find_last_of('\\');
            if (pos == std::string::npos)
#endif
                return false;
            if (!createFolders(folder.substr(0, pos)))
                return false;
            return 0 == makeDir(folder);
        }
        case EEXIST:
            return folderExists(folder);

        default:
            return false;
    }
}

void setAutoSubfolder(std::string &recordingFolder) {
    auto now = std::chrono::system_clock::now();
    auto timePoint = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&timePoint);
    std::ostringstream oss;
    oss << recordingFolder;
    oss << std::put_time(&localTime, "/%Y-%m-%d_%H-%M-%S");
    recordingFolder = oss.str();
}
