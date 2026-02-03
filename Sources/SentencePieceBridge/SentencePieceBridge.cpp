//
//  SentencePieceBridge-Framework.cpp
//  C++ implementation of the bridge for XCFramework usage
//

#include "SentencePieceBridge.h"

// When using XCFramework, we need to include the framework header
#ifdef __has_include
  #if __has_include(<SentencePiece/sentencepiece_processor.h>)
    #include <SentencePiece/sentencepiece_processor.h>
  #elif __has_include("sentencepiece_processor.h")
    #include "sentencepiece_processor.h"
  #else
    #include <sentencepiece_processor.h>
  #endif
#else
  #include <sentencepiece_processor.h>
#endif

#include <vector>
#include <string>
#include <cstring>

extern "C" {

SentencePieceProcessor sentencepiece_create(const char* model_path) {
    auto* processor = new sentencepiece::SentencePieceProcessor();
    const auto status = processor->Load(model_path);
    if (!status.ok()) {
        delete processor;
        return nullptr;
    }
    return processor;
}

int sentencepiece_encode_as_pieces(SentencePieceProcessor processor,
                                  const char* text,
                                  char*** pieces) {
    if (!processor || !text || !pieces) return 0;

    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    std::vector<std::string> pieces_vec;

    const auto status = sp->Encode(text, &pieces_vec);
    if (!status.ok()) return 0;

    // Allocate array of C strings
    *pieces = (char**)malloc(pieces_vec.size() * sizeof(char*));
    for (size_t i = 0; i < pieces_vec.size(); ++i) {
        (*pieces)[i] = strdup(pieces_vec[i].c_str());
    }

    return pieces_vec.size();
}

int sentencepiece_encode_as_ids(SentencePieceProcessor processor,
                               const char* text,
                               int** ids) {
    if (!processor || !text || !ids) return 0;

    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    std::vector<int> ids_vec;

    const auto status = sp->Encode(text, &ids_vec);
    if (!status.ok()) return 0;

    // Allocate array of ints
    *ids = (int*)malloc(ids_vec.size() * sizeof(int));
    memcpy(*ids, ids_vec.data(), ids_vec.size() * sizeof(int));

    return ids_vec.size();
}

int sentencepiece_get_piece_size(SentencePieceProcessor processor) {
    if (!processor) return 0;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->GetPieceSize();
}

int sentencepiece_piece_to_id(SentencePieceProcessor processor, const char* piece) {
    if (!processor || !piece) return -1;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->PieceToId(piece);
}

const char* sentencepiece_id_to_piece(SentencePieceProcessor processor, int id) {
    if (!processor) return nullptr;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    static thread_local std::string piece;
    piece = sp->IdToPiece(id);
    return piece.c_str();
}

float sentencepiece_get_score(SentencePieceProcessor processor, int id) {
    if (!processor) return 0.0f;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->GetScore(id);
}

char* sentencepiece_decode_ids(SentencePieceProcessor processor, const int* ids, int num_ids) {
    if (!processor || !ids || num_ids <= 0) return nullptr;

    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    std::vector<int> ids_vec(ids, ids + num_ids);
    std::string decoded;

    const auto status = sp->Decode(ids_vec, &decoded);
    if (!status.ok()) return nullptr;

    return strdup(decoded.c_str());
}

void sentencepiece_destroy(SentencePieceProcessor processor) {
    if (processor) {
        delete static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    }
}

void sentencepiece_free_processor(SentencePieceProcessor processor) {
    if (processor) {
        delete static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    }
}

void sentencepiece_free_pieces(char** pieces, int count) {
    if (pieces) {
        for (int i = 0; i < count; ++i) {
            free(pieces[i]);
        }
        free(pieces);
    }
}

void sentencepiece_free_ids(int* ids) {
    if (ids) {
        free(ids);
    }
}

} // extern "C"
