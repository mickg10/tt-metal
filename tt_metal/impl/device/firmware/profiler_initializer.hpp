// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "firmware_initializer.hpp"

namespace tt::tt_metal {

struct ProfilerStateManager;

class ProfilerInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key() { return InitializerKey::Profiler; }

    ProfilerInitializer(
        const Hal& hal,
        Cluster& cluster,
        const llrt::RunTimeOptions& rtoptions,
        std::shared_ptr<const ContextDescriptor> descriptor,
        bool skip_remote_devices,
        ProfilerStateManager* profiler_state_manager);

    void init(const std::vector<IDevice*>& devices) override;
    void configure() override;
    void teardown() override;
    bool is_initialized() const override;

private:
    [[maybe_unused]] bool skip_remote_devices_;
    [[maybe_unused]] ProfilerStateManager* profiler_state_manager_;
    std::vector<IDevice*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
