import os
import sys

# Define os marcadores de início e fim do bloco do footer a ser removido
# O marcador inicial é o comentário que você especificou.
START_MARKER = "<!-- Footer Adicionado pelo Script -->"
# O marcador final é a tag de fechamento do footer.
END_MARKER = "</footer>"

def remove_footer_from_html(filepath):
    """
    Lê um arquivo HTML, procura pelo bloco de footer específico,
    remove-o se encontrado e salva o arquivo. Retorna True se modificado, False caso contrário.
    """
    filename = os.path.basename(filepath)
    try:
        # Tenta ler o arquivo com encoding UTF-8 (mais comum na web)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            original_encoding = 'utf-8'
    except UnicodeDecodeError:
        try:
            # Se falhar, tenta com Latin-1 (comum no Windows)
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
                original_encoding = 'cp1252'
        except Exception as e:
            print(f"Erro ao ler '{filename}': Não foi possível decodificar o arquivo ({e}). Pulando.")
            return False
    except IOError as e:
        print(f"Erro de I/O ao ler '{filename}': {e}. Pulando.")
        return False

    original_content = content
    modified = False

    # Procura pelo bloco repetidamente (caso haja mais de um, embora improvável)
    while True:
        start_index = content.find(START_MARKER)

        # Se não encontrar o marcador inicial, para a busca neste arquivo
        if start_index == -1:
            break

        # Se encontrou o marcador inicial, procura o marcador final *depois* dele
        end_index = content.find(END_MARKER, start_index)

        # Se não encontrar o marcador final correspondente, é um problema.
        # Imprime um aviso e para de modificar este arquivo para segurança.
        if end_index == -1:
            print(f"AVISO: Marcador inicial encontrado, mas tag '{END_MARKER}' não encontrada depois dele em '{filename}'. Nenhuma remoção feita a partir deste ponto no arquivo.")
            break

        # Calcula a posição exata do fim do bloco (incluindo a tag END_MARKER)
        end_pos = end_index + len(END_MARKER)

        # Remove o bloco inteiro (do início do comentário até o fim do footer)
        # Mantém o que veio ANTES do marcador inicial e o que veio DEPOIS do marcador final.
        content = content[:start_index] + content[end_pos:]
        modified = True
        # Continua o loop para o caso de haver mais blocos *idênticos* a serem removidos

    # Se o conteúdo foi modificado, salva o arquivo
    if modified:
        try:
            # Salva o arquivo com o mesmo encoding que foi lido, se possível,
            # ou UTF-8 como padrão ao escrever. UTF-8 é geralmente seguro.
            write_encoding = original_encoding if original_encoding else 'utf-8'
            with open(filepath, 'w', encoding=write_encoding) as f:
                f.write(content)
            print(f"Footer removido com sucesso de: '{filename}'")
            return True
        except IOError as e:
            print(f"Erro de I/O ao salvar alterações em '{filename}': {e}")
            # Opcional: Poderia tentar reverter para o conteúdo original aqui,
            # mas pode ser complexo. Melhor apenas reportar o erro.
            return False
        except Exception as e:
            print(f"Erro inesperado ao salvar alterações em '{filename}': {e}")
            return False
    else:
        # Informa que o footer não foi encontrado (ou já tinha sido removido)
        # print(f"Nenhum bloco de footer correspondente encontrado em: '{filename}'")
        return False # Retorna False pois não houve modificação

def main():
    # Obtém o diretório onde o script está sendo executado
    # Assume-se que os arquivos HTML estão neste mesmo diretório ("raiz")
    current_dir = os.getcwd()
    print(f"Procurando arquivos HTML em: {current_dir}")
    print("-" * 30)

    processed_files = 0
    modified_files = 0

    # Lista todos os arquivos no diretório atual
    for filename in os.listdir(current_dir):
        # Verifica se é um arquivo HTML
        if filename.lower().endswith(('.html', '.htm')):
            processed_files += 1
            filepath = os.path.join(current_dir, filename)
            # Chama a função para processar o arquivo
            if remove_footer_from_html(filepath):
                modified_files += 1

    print("-" * 30)
    if processed_files == 0:
        print("Nenhum arquivo .html ou .htm encontrado no diretório.")
    else:
        print(f"Processamento concluído.")
        print(f"Total de arquivos HTML/HTM encontrados: {processed_files}")
        print(f"Arquivos modificados: {modified_files}")
        print(f"Arquivos não modificados (footer não encontrado ou erro): {processed_files - modified_files}")

if __name__ == "__main__":
    main()