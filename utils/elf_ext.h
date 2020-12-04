#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <elf.h>
#include <link.h>

#pragma once

bool elf_get_ident(FILE * file, unsigned char * e_ident);

unsigned char elf_get_class(FILE * file); 

bool elf_is_elf64(FILE * file); 

bool elf32_get_elf_header(FILE * file, Elf32_Ehdr * elf_header);

bool elf64_get_elf_header(FILE * file, Elf64_Ehdr * elf_header);

bool elf32_get_section_header(FILE * file, const Elf32_Ehdr * elf_header, unsigned int index,
		                                   Elf32_Shdr * header);

bool elf64_get_section_header(FILE * file, const Elf64_Ehdr * elf_header, unsigned int index,
		                                   Elf64_Shdr * header); 

char * elf32_get_string(FILE * file, const Elf32_Ehdr * elf_header, unsigned int offset);

char * elf64_get_string(FILE * file, const Elf64_Ehdr * elf_header, unsigned int offset);

bool elf32_get_section_header_by_name(FILE * file, const Elf32_Ehdr * elf_header, const char * name,
		        Elf32_Shdr * header);

bool elf64_get_section_header_by_name(FILE * file, const Elf64_Ehdr * elf_header, const char * name,
		        Elf64_Shdr * header);

bool elf32_get_dynamic_section(FILE * file, const Elf32_Ehdr * elf_header, Elf32_Shdr * header);

bool elf64_get_dynamic_section(FILE * file, const Elf64_Ehdr * elf_header, Elf64_Shdr * header);

bool elf32_get_strtab_section(FILE * file, const Elf32_Ehdr * elf_header, Elf32_Shdr * header);

bool elf64_get_strtab_section(FILE * file, const Elf64_Ehdr * elf_header, Elf64_Shdr * header); 

