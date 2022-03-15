AUTOMAKE_OPTIONS = subdir-objects
include $(top_srcdir)/atrip.mk

AM_CXXFLAGS = $(CTF_CPPFLAGS)

lib_LIBRARIES = libatrip.a

libatrip_a_CPPFLAGS = -I$(top_srcdir)/include/
libatrip_a_SOURCES = ./atrip/Atrip.cxx
