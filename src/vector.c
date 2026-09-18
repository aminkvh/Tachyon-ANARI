/*
 * vector.c - vector arithmetic functions.
 *
 *  $Id: vector.c,v 1.8 2012/10/17 04:25:57 johns Exp $
 *
 * These used to be defined here as ordinary externally-linked functions.
 * They're tiny, allocation-free leaf functions called tens of millions of
 * times per render, so the cross-TU call overhead mattered; they're now
 * `static inline` directly in vector.h so every call site can fold them
 * in. This file is kept (compiling to an empty translation unit) so the
 * legacy unix/Makefile's vector.o build rule still resolves.
 */
