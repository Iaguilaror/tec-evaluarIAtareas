# load packages
pacman::p_load( "pdftools", "quanteda", "quanteda.textstats",
                "dplyr", "ggplot2", "proxyC", "stringr",
                "igraph", "ggraph" )

# list all pdf
allpdf.v <- list.files( path = "submissions/", pattern = ".pdf", full.names = T )

str(allpdf.v)

# Analyze by similarity ----
# 1. Read text from the PDFs
# pdftools::pdf_text reads page by page, so we collapse pages into a single string per document
raw_texts <- lapply ( X = allpdf.v, FUN = function ( file_path ) {
  pages <- pdf_text ( pdf = file_path )
  full_text <- paste ( x = pages, collapse = " " )
  return ( full_text )
} )

# Build a dataframe mapping the filenames to their extracted text
essay_data <- data.frame ( 
  doc_id = allpdf.v, 
  text = unlist ( x = raw_texts ), 
  stringsAsFactors = FALSE ) %>% 
  mutate( doc_id = doc_id %>% str_remove( "submissions/" ) %>% str_remove("_.*") )

# 2. Build the corpus and tokenize
essay_corpus <- corpus ( x = essay_data, docid_field = "doc_id", text_field = "text" )

# Tokenize text, stripping punctuation, numbers, and symbols to focus strictly on vocabulary
essay_tokens <- tokens ( 
  x = essay_corpus, 
  remove_punct = TRUE, 
  remove_numbers = TRUE, 
  remove_symbols = TRUE 
)

# 3. Create a Document-Feature Matrix (DFM) and apply TF-IDF weighting
essay_dfm <- dfm ( x = essay_tokens )
essay_tfidf <- dfm_tfidf ( x = essay_dfm )

# 4. Calculate pairwise cosine similarity
sim_matrix <- textstat_simil ( x = essay_tfidf, method = "cosine",
                               margin = "documents" )

# 5. Convert to a dataframe to filter for the highest similarities
all_sim_df <- as.data.frame ( x = sim_matrix )  %>% 
  mutate( document1 = as.character( document1 ),
          document2 = as.character( document2 ) )

sim_df <- all_sim_df %>%
  rename ( similarity = cosine ) %>%
  # Remove self-comparisons and duplicate directional pairs
  filter ( document1 != document2 ) %>%
  arrange ( desc ( x = similarity ) )

# 6. Plot a heatmap to visually flag clustered submissions
ggplot ( data = sim_df,
         mapping = aes ( x = document1,
                         y = document2,
                         fill = similarity,
                         label = round(similarity, 2) ) ) +
  geom_tile ( ) +
  geom_text( color = "white" ) +
  scale_fill_viridis_c ( option = "plasma", limits = c ( 0, 1 ) ) +
  theme_minimal ( ) +
  theme ( 
    axis.text.x = element_text ( angle = 90, hjust = 1, size = 6 ),
    axis.text.y = element_text ( size = 6 ) ) +
  labs ( title = "Cosine Similarity Between Documents",
         x = "Document A", y = "Document B", fill = "Similarity" )

# Ver como red ----
# 1. Filtrar las relaciones significativas
# Usamos el dataframe 'sim_df' que ya tienes del paso anterior.
# Filtramos para quedarnos solo con las similitudes mayores a 0.10
edges_filtered <- filter ( .data = sim_df, similarity > 0.10 )

# 2. Crear el objeto de red (grafo no dirigido)
essay_network <- graph_from_data_frame ( d = edges_filtered, directed = FALSE )

# 3. Graficar la red
# Usamos el algoritmo de diseño "fr" (Fruchterman-Reingold) que dispersa bien los nodos
ggraph ( graph = essay_network, layout = "fr" ) +
  # Dibujar las líneas de conexión; el grosor y la transparencia dependerán del nivel de similitud
  geom_edge_link ( 
    mapping = aes ( width = similarity, alpha = similarity ), 
    color = "darkblue" 
  ) +
  # Dibujar los nodos (cada nodo es un ensayo/estudiante)
  geom_node_point ( 
    color = "orange", 
    size = 4 
  ) +
  # Agregar los nombres de los estudiantes a cada nodo, usando repel para que no se encimen
  geom_node_text ( 
    mapping = aes ( label = name ), 
    repel = TRUE, 
    size = 3 
  ) +
  # Ajustar la escala del grosor de las líneas
  scale_edge_width ( range = c ( 0.5, 2.5 ) ) +
  # Limpiar el fondo para que luzca como una red pura
  theme_void ( ) +
  labs ( 
    title = "Red de Similitud Léxica entre Ensayos", 
    subtitle = "Se muestran únicamente los pares con similitud de coseno > 0.10",
    edge_width = "Similitud",
    edge_alpha = "Similitud"
  )

# Try perplexity
library ( package = "reticulate" )

# 1. Create a dedicated virtual environment
virtualenv_create ( envname = "nlp_env" )

# 2. Install the modules specifically into this new environment
virtualenv_install ( envname = "nlp_env", packages = c ( "transformers", "torch" ) )

# 3. Force R to use this exact environment
use_virtualenv ( virtualenv = "nlp_env", required = TRUE )

# 4. Import the modules
transformers <- import ( module = "transformers" )
torch <- import ( module = "torch" )
library ( package = "dplyr" )

# Importar los módulos de Python
transformers <- import ( module = "transformers" )
torch <- import ( module = "torch" )

# Cargar un modelo ligero entrenado en español para evaluar la predictibilidad del texto
model_name <- "DeepESP/gpt2-spanish"
tokenizer <- transformers$AutoTokenizer$from_pretrained ( pretrained_model_name_or_path = model_name )
model <- transformers$AutoModelForCausalLM$from_pretrained ( pretrained_model_name_or_path = model_name )

# Definir la función para calcular la perplejidad
calculate_perplexity <- function ( text_input ) {
  
  # Tokenizar el texto, limitando a 512 tokens para evitar saturación de memoria
  inputs <- tokenizer ( text = text_input, return_tensors = "pt", truncation = TRUE, max_length = 512L )
  
  # Desactivar el cálculo de gradientes para acelerar el proceso
  torch$set_grad_enabled ( mode = FALSE )
  
  # Pasar los tokens por el modelo
  outputs <- model ( input_ids = inputs$input_ids, labels = inputs$input_ids )
  loss <- outputs$loss
  
  # Calcular la perplejidad como la función exponencial de la pérdida
  perplexity_tensor <- torch$exp ( input = loss )
  
  # Reactivar gradientes
  torch$set_grad_enabled ( mode = TRUE )
  
  return ( perplexity_tensor$item ( ) )
}

# Aplicar la función al dataframe essay_data que ya contiene los textos extraídos
perplexity_results <- essay_data %>%
  rowwise ( data = . ) %>%
  mutate ( .data = ., perplexity_score = calculate_perplexity ( text_input = text ) ) %>%
  select ( .data = ., doc_id, perplexity_score ) %>%
  arrange ( .data = ., perplexity_score )

perplexity_results_final <- perplexity_results %>% 
  mutate( tag = case_when( perplexity_score < 25 ~ "alto uso de IA",
                           perplexity_score >= 100 ~ "poco uso de IA",
                           TRUE ~ "punto medio" ) )

mycolors.v <- c( "alto uso de IA" = "tomato",
                 "punto medio" =  "skyblue",
                 "poco uso de IA" = "limegreen" )


# plot perplexity results
ggplot( data = perplexity_results_final,
        mapping = aes( x = doc_id,
                       y = perplexity_score,
                       fill = tag ) ) +
  geom_col( color = "black" ) +
  geom_hline( yintercept = 25,
               lty = "dashed", color = "tomato" ) +
  geom_hline( yintercept = 100,
              lty = "dotted", color = "limegreen" ) +
  scale_x_discrete( limits = perplexity_results$doc_id ) +
  scale_fill_manual( values = mycolors.v ) +
  theme_classic( ) +
  theme( axis.text.x = element_text( angle = 90 ) )
