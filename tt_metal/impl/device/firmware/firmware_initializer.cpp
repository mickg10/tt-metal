// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "firmware_initializer.hpp"

#include "impl/context/context_descriptor.hpp"

namespace tt::tt_metal {

FirmwareInitializer::FirmwareInitializer(
    const Hal& hal,
    Cluster& cluster,
    const llrt::RunTimeOptions& rtoptions,
    std::shared_ptr<const ContextDescriptor> descriptor) :
    hal_(hal), cluster_(cluster), rtoptions_(rtoptions), descriptor_(std::move(descriptor)) {}

}  // namespace tt::tt_metal
