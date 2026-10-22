// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "TachyonGlobalState.h"
// helium
#include "helium/BaseObject.h"
#include "helium/array/Array1D.h"
#include "helium/array/ObjectArray.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>

namespace tachyon_anari {

using Array1D = helium::Array1D;
using ObjectArray = helium::ObjectArray;

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, TachyonGlobalState *s);
  virtual ~Object() = default;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;

  bool isValid() const override;

  // Objects that change the Tachyon scene contents return true, which makes
  // every frame rebuild its Tachyon scene after they are (re)finalized.
  virtual bool affectsScene() const;

  TachyonGlobalState *deviceState() const;
};

// Placeholder for object subtypes this device does not implement. Created
// instead of failing so applications keep running; skipped at render time.
template <typename BASE>
struct Unsupported : public BASE
{
  Unsupported(TachyonGlobalState *s) : BASE(s) {}
  bool isValid() const override { return false; }
};

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Object *, ANARI_OBJECT);
}
