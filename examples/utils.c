#include "utils.h"
#include <sys/stat.h>

// Function to find the correct path to the data directory
const char* find_data_path_prefix() {
    struct stat st;
    // Check if "data" directory exists in the current directory
    if (stat("data", &st) == 0 && S_ISDIR(st.st_mode)) {
        return "data/";
    }
    // Check if "data" directory exists in the parent directory
    if (stat("../data", &st) == 0 && S_ISDIR(st.st_mode)) {
        return "../data/";
    }
    // Default to current directory if not found elsewhere
    return "data/";
}
