// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "firmware_initializer.hpp"

namespace tt::tt_metal {

class CommandQueueInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key() { return InitializerKey::CommandQueue; }

    CommandQueueInitializer(
        const Hal& hal,
        Cluster& cluster,
        const llrt::RunTimeOptions& rtoptions,
        std::shared_ptr<const ContextDescriptor> descriptor,
        bool skip_remote_devices);

    void init(const std::vector<IDevice*>& devices) override;
    void configure() override;
    void teardown() override;
    bool is_initialized() const override;

private:
    void initialize_host(IDevice* dev) const;

    bool using_fast_dispatch() const;

    bool skip_remote_devices_;
    std::vector<IDevice*> devices_;
    bool initialized_ = false;
};

}  // namespace tt::tt_metal
