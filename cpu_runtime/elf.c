#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <elf.h>
#include <link.h>


static bool elf_get_ident(FILE * file, unsigned char * e_ident) {
	    if (fseek(file, 0, SEEK_SET))
		            return false;
	        if (fread(e_ident, EI_NIDENT, 1, file) != 1)
			        return false;
		    return (strncmp((char *) e_ident, ELFMAG, SELFMAG) == 0);
}

static unsigned char elf_get_class(FILE * file) {
	    unsigned char e_ident[EI_NIDENT];
	        if (!elf_get_ident(file, e_ident))
			        return '\0';
		    return e_ident[EI_CLASS];
}

bool elf_is_elf64(FILE * file) {
	    return (elf_get_class(file) == ELFCLASS64);
}

static bool elf32_get_elf_header(FILE * file, Elf32_Ehdr * elf_header) {
	    if (fseek(file, 0, SEEK_SET))
		            return false;
	        if (fread(elf_header, sizeof(Elf32_Ehdr), 1, file) != 1)
			        return false;
		    return (strncmp((char *) &elf_header->e_ident, ELFMAG, SELFMAG) == 0);
}

static bool elf64_get_elf_header(FILE * file, Elf64_Ehdr * elf_header) {
	    if (fseek(file, 0, SEEK_SET))
		            return false;
	        if (fread(elf_header, sizeof(Elf64_Ehdr), 1, file) != 1)
			        return false;
		    return (strncmp((char *) &elf_header->e_ident, ELFMAG, SELFMAG) == 0);
}

static bool elf32_get_section_header(FILE * file, const Elf32_Ehdr * elf_header, unsigned int index,
		                                   Elf32_Shdr * header) {
	    unsigned int offset = elf_header->e_shoff + elf_header->e_shentsize * index;
	        if (fseek(file, offset, SEEK_SET))
			        return false;
		    return (fread(header, sizeof(Elf32_Shdr), 1, file) == 1);
}

static bool elf64_get_section_header(FILE * file, const Elf64_Ehdr * elf_header, unsigned int index,
		                                   Elf64_Shdr * header) {
	    unsigned int offset = elf_header->e_shoff + elf_header->e_shentsize * index;
	        if (fseek(file, offset, SEEK_SET))
			        return false;
		    return (fread(header, sizeof(Elf64_Shdr), 1, file) == 1);
}

static char * elf32_get_string(FILE * file, const Elf32_Ehdr * elf_header, unsigned int offset) {
	    Elf32_Shdr sec_head;
	        if (!elf32_get_section_header(file, elf_header, elf_header->e_shstrndx, &sec_head))
			        return NULL;
		        
		    char buf[256];
		        
		        offset += sec_head.sh_offset;
			    if (fseek(file, offset, SEEK_SET))
				            return NULL;
			            
			        size_t got = fread(buf, 1, 255, file);
				    
				    return strndup(buf, got);
}

static char * elf64_get_string(FILE * file, const Elf64_Ehdr * elf_header, unsigned int offset) {
	    Elf64_Shdr sec_head;
	        if (!elf64_get_section_header(file, elf_header, elf_header->e_shstrndx, &sec_head))
			        return NULL;

		    char buf[256];

		        offset += sec_head.sh_offset;
			    if (fseek(file, offset, SEEK_SET))
				            return NULL;

			        size_t got = fread(buf, 1, 255, file);

				    return strndup(buf, got);
}

static bool elf32_get_section_header_by_name(FILE * file, const Elf32_Ehdr * elf_header, const char * name,
		        Elf32_Shdr * header) {
	    bool ret = false;
	        for (int i = 0; (i < elf_header->e_shnum) && (!ret); ++i) {
			        if (!elf32_get_section_header(file, elf_header, i, header))
					            return false;
				        char * section_name = elf32_get_string(file, elf_header, header->sh_name);
					        ret = (strcmp(name, section_name) == 0);
						        free(section_name);
							    }
		    return ret;
}

static bool elf64_get_section_header_by_name(FILE * file, const Elf64_Ehdr * elf_header, const char * name,
		        Elf64_Shdr * header) {
	    bool ret = false;
	        for (int i = 0; (i < elf_header->e_shnum) && (!ret); ++i) {
			        if (!elf64_get_section_header(file, elf_header, i, header))
					            return false;
				        char * section_name = elf64_get_string(file, elf_header, header->sh_name);
					        ret = (strcmp(name, section_name) == 0);
						        free(section_name);
							    }
		    return ret;
}

static bool elf32_get_dynamic_section(FILE * file, const Elf32_Ehdr * elf_header, Elf32_Shdr * header) {
	    if (!elf32_get_section_header_by_name(file, elf_header, ".dynamic", header))
		            return false;
	        return header->sh_type == SHT_DYNAMIC;
}

static bool elf64_get_dynamic_section(FILE * file, const Elf64_Ehdr * elf_header, Elf64_Shdr * header) {
	    if (!elf64_get_section_header_by_name(file, elf_header, ".dynamic", header))
		            return false;
	        return header->sh_type == SHT_DYNAMIC;
}

static bool elf32_get_symtab_section(FILE * file, const Elf32_Ehdr * elf_header, Elf32_Shdr * header) {
            if (!elf32_get_section_header_by_name(file, elf_header, ".symtab", header))
                            return false;
                return header->sh_type == SHT_SYMTAB;
}

static bool elf64_get_symtab_section(FILE * file, const Elf64_Ehdr * elf_header, Elf64_Shdr * header) {
            if (!elf64_get_section_header_by_name(file, elf_header, ".symtab", header))
                            return false;
                return header->sh_type == SHT_SYMTAB;
}

