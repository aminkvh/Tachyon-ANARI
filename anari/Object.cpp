// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "Object.h"

namespace tachyon_anari {

Object::Object(ANARIDataType type, TachyonGlobalState *s)
    : helium::BaseObject(type, s)
{}

void Object::commitParameters() {}

void Object::finalize() {}

void Object::markFinalized()
{
  helium::BaseObject::markFinalized();
  if (affectsScene())
    deviceState()->sceneStamp = helium::newTimeStamp();
}

bool Object::affectsScene() const
{
  return false;
}

bool Object::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t,
    uint32_t)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }
  return false;
}

bool Object::isValid() const
{
  return true;
}

TachyonGlobalState *Object::deviceState() const
{
  return static_cast<TachyonGlobalState *>(helium::BaseObject::m_state);
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Object *);
}
