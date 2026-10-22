// Copyright 2023-2026 The Khronos Group
// Copyright (c) 2026 Tachyon-dev contributors (adapted from the helide device)
// SPDX-License-Identifier: Apache-2.0

#include "TachyonDevice.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_tachyon_export.h"

namespace tachyon_anari {

const char **query_extensions();

struct TachyonLibrary : public anari::LibraryImpl
{
  TachyonLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

TachyonLibrary::TachyonLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice TachyonLibrary::newDevice(const char * /*subtype*/)
{
  // only the "default" device subtype exists
  return (ANARIDevice) new TachyonDevice(this_library());
}

const char **TachyonLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace tachyon_anari

// Define library entrypoint //////////////////////////////////////////////////

extern "C" TACHYON_ANARI_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    tachyon, handle, scb, scbPtr)
{
  return (ANARILibrary) new tachyon_anari::TachyonLibrary(handle, scb, scbPtr);
}

extern "C" TACHYON_ANARI_EXPORT ANARIDevice anariNewTachyonDevice(
    ANARIStatusCallback defaultCallback, const void *userPtr)
{
  return (ANARIDevice) new tachyon_anari::TachyonDevice(defaultCallback, userPtr);
}
