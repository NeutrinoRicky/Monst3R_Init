/*
 * Minimal runtime shim for environments missing Intel ITT notify shared libs.
 *
 * PyTorch may link against iJIT_* symbols through oneDNN/MKL builds. When the
 * dynamic lib providing those symbols is unavailable, importing torch fails.
 * This shim provides no-op implementations so torch can initialize.
 */

#include <stdint.h>

int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

uint32_t iJIT_GetNewMethodID(void) {
    static uint32_t next_id = 1;
    return next_id++;
}

