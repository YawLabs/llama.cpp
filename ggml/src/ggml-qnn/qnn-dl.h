#pragma once

// dynamic library loading shared by qnn-lib.cpp (QnnHtp) and qnn-mem.cpp (libcdsprpc)

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#include <cstring>
#include <string>

// a path with a directory separator is loaded with LOAD_WITH_ALTERED_SEARCH_PATH so its
// dependent DLLs resolve from the same directory, a bare name goes through the default search
static inline void * ggml_qnn_dl_open(const char * path) {
#ifdef _WIN32
    if (strchr(path, '\\') || strchr(path, '/')) {
        return (void *) LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

// why the last ggml_qnn_dl_open failed. call it right after the failed load, before anything
// else can overwrite the thread's last error: "not found" and "exists but a dependent DLL is
// missing" (126) or "wrong architecture" (193) need different fixes
static inline std::string ggml_qnn_dl_error(void) {
#ifdef _WIN32
    const DWORD code = GetLastError();
    char msg[256] = "";
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, msg, sizeof(msg), NULL);
    while (n > 0 && (msg[n - 1] == '\r' || msg[n - 1] == '\n' || msg[n - 1] == ' ' || msg[n - 1] == '.')) {
        msg[--n] = 0;
    }
    return "error " + std::to_string((unsigned long) code) + (n ? std::string(": ") + msg : std::string());
#else
    const char * err = dlerror();
    return err ? err : "unknown error";
#endif
}

static inline void * ggml_qnn_dl_sym(void * lib, const char * name) {
#ifdef _WIN32
    return (void *) GetProcAddress((HMODULE) lib, name);
#else
    return dlsym(lib, name);
#endif
}

static inline void ggml_qnn_dl_close(void * lib) {
#ifdef _WIN32
    FreeLibrary((HMODULE) lib);
#else
    dlclose(lib);
#endif
}
