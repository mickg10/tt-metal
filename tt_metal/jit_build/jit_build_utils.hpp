// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/stable_hash.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);

// An RAII wrapper that generates a temporary filename and renames the file on destruction.
// This is to allow multiple processes to write to the same target file without clobbering each other.
class FileRenamer {
public:
    FileRenamer(const std::string& target_path);
    FileRenamer(const FileRenamer&) = delete;
    FileRenamer& operator=(const FileRenamer&) = delete;
    FileRenamer(FileRenamer&&) = default;
    FileRenamer& operator=(FileRenamer&&) = default;
    ~FileRenamer();

    const std::string& path() const { return temp_path_; }

private:
    std::string temp_path_;
    std::string target_path_;
    static uint64_t unique_id_;
};

// An RAII wrapper that keeps track of a group of files, some of which need to be renamed.
class FileGroupRenamer {
public:
    FileGroupRenamer(std::vector<std::string> target_paths) : paths_(std::move(target_paths)) {}
    std::string& generate_temp_path(size_t i) {
        renamers_.emplace_back(paths_[i]);
        paths_[i] = renamers_.back().path();
        return paths_[i];
    }
    // Returns the temp path or the original path if no temp path was generated.
    const std::vector<std::string>& paths() const { return paths_; }

private:
    std::vector<std::string> paths_;
    std::vector<FileRenamer> renamers_;
};

}  // namespace tt::jit_build::utils
