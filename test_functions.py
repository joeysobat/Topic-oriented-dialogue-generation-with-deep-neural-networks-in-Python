
import torch
import torch.nn as nn
import torch.nn.functional as F
import Data_processing as dpros
from torch.autograd import Variable
import numpy as np
import math
import rouge
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import statistics


nltk.download('punkt')

pad_len = 30



## this function outputs the corresponding indices of the words when calculating
## the word embedding similarity:

def zero_pad_test(sentence, w2idx):

    idx_q = np.zeros([dpros.limit['maxq']], dtype=np.int64)

    a_indices = pad_seq_test(sentence, w2idx)
    idx_q = np.array(a_indices)

    return idx_q
	



## this function converts the words to indices when calculating the word embedding
## similarity:

def pad_seq_test(seq, lookup):

    indices = []
    
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices
	
	


## this function predicts the responses for the source sentences of the test dataset
## for the emotion embedding model:

def light_predict(model, source_batch, seq_length_batch, word2idx):
    
    model.eval()
    source_batch = source_batch.cuda()
    seq_length_batch = seq_length_batch.cuda()

    final_outputs = list()

    for i in range(dpros.batch_size):

      source = source_batch[i]
      seq_length = seq_length_batch[i]

      source = source.unsqueeze(0)
      seq_length = seq_length.unsqueeze(0)
      
      source = model.embeddings(source)

      src_output, (src_hidden, src_cell) = model.encoder(source, seq_length)
      
      context = src_output.transpose(0, 1)
      
      outputs = torch.zeros(pad_len, dtype=torch.int64).cuda()
      outputs[0] = torch.LongTensor([word2idx['<s>']])

      dec_states = [src_hidden, src_cell]
      
      decoder_init_state = nn.Tanh()(model.encoder2decoder(dec_states[0])).cuda()
      
      dec_states[0] = decoder_init_state

      for j in range(1, pad_len):
      
          trg_input = model.embeddings(outputs[:j].unsqueeze(0))
          
          out, (trg_hidden, trg_cell) = model.decoder(trg_input, (dec_states[0], dec_states[1]), context)
          
          dec_states = [trg_hidden, trg_cell]
          out_linear = model.out(out)
          out_softmax = F.softmax(out_linear, dim=-1)
          val, ix = out_softmax[:, -1].data.topk(1)
          outputs[j] = ix[0][0]
          if ix[0][0] == word2idx['</s>']:
              break

      final_outputs.append(outputs)
    sentence_outputs = torch.stack(final_outputs)
    
    return sentence_outputs



## this function predicts the responses for the source sentences of the test dataset
## for the transformer model:

def light_predict_trans(model, source_batch, word2id):
    
  model.eval()
  source_batch = source_batch.cuda()
  target_pad = word2id['<pad>']

  final_outputs = list()

  for i in range(dpros.batch_size):

    source = source_batch[i]
    
    source = source.unsqueeze(0)
    
    src_mask = (source != target_pad).unsqueeze(-2)
    
    source_embedding = model.embedding(source)
  
    enc_sentence = model.encoder(source_embedding, src_mask)

    outputs = torch.zeros(pad_len, dtype=torch.int64).cuda()
    outputs[0] = torch.LongTensor([word2id['<s>']])

    for j in range(1, pad_len):
        
        trg_input = outputs[:j].unsqueeze(0)
        
        trg_input_embedding = model.embedding(trg_input)
        
        target_mask = (trg_input != target_pad).unsqueeze(-2).cuda()

        nopeak_mask = np.triu(np.ones((1, j, j)), k=1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).cuda()
        trg_mask = target_mask & nopeak_mask

        dec_sentence = model.decoder(trg_input_embedding, enc_sentence, src_mask, trg_mask)

        out = model.linear_out(dec_sentence)
        out_softmax = F.softmax(out, dim=-1)
        val, ix = out_softmax[:, -1].data.topk(1)
        outputs[j] = ix[0][0]
        if ix[0][0] == word2id['</s>']:
            break

    final_outputs.append(outputs)
  sentence_outputs = torch.stack(final_outputs)

  return final_outputs
  
  


## this function prepares the format for showing the results of the Rouge metric:

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
  
  

## this function makes the predictions for the test dataset and converts the indices back to words.
## It outputs the predictions and the target answers:

def create_answers_preds(light_model, test_loader, word2id, id2word):

    answers = list()
    predict = list()

    light_model.freeze()

    for i, data in enumerate(test_loader, 0):
        
        src, src_length, trg = data

        predicted = light_predict(light_model.cuda(), src, src_length, word2id)

        for elem in trg:
            real = dpros.decode(sequence=elem.numpy(), lookup=id2word, separator=' ')
            answers.append(real)

        for elem2 in predicted:
            fake = dpros.decode(sequence=elem2.cpu().numpy(), lookup=id2word, separator=' ')
            predict.append(fake)
            
    return answers, predict
    
    
    
## this function makes the predictions for the test dataset and converts the indices back to words.
## It outputs the predictions and the source sentences:
    
def create_sources_preds(light_model, test_loader, word2id, id2word):

    sources = list()
    predict = list()

    light_model.freeze()

    for i, data in enumerate(test_loader, 0):
        
        src, src_length, trg = data

        predicted = light_predict(light_model.cuda(), src, src_length, word2id)

        for elem in src:
            real = dpros.decode(sequence=elem.numpy(), lookup=id2word, separator=' ')
            sources.append(real)

        for elem2 in predicted:
            fake = dpros.decode(sequence=elem2.cpu().numpy(), lookup=id2word, separator=' ')
            predict.append(fake)
            
    return sources, predict
    
    
    

## the same as "create_answers_preds", but for the transformer model:

def create_answers_preds_trans(light_model, test_loader, word2id, id2word):

    answers = list()
    predict = list()

    light_model.freeze()

    for i, data in enumerate(test_loader, 0):
        
        src, trg = data

        predicted = light_predict_trans(light_model.cuda(), src, word2id)

        for elem in trg:
            real = dpros.decode(sequence=elem.numpy(), lookup=id2word, separator=' ')
            answers.append(real)

        for elem2 in predicted:
            fake = dpros.decode(sequence=elem2.cpu().numpy(), lookup=id2word, separator=' ')
            predict.append(fake)
            
    return answers, predict
    
    


## the same as "create_sources_preds", but for the transformer model:

def create_sources_preds_trans(light_model, test_loader, word2id, id2word):

    sources = list()
    predict = list()

    light_model.freeze()

    for i, data in enumerate(test_loader, 0):
        
        src, trg = data

        predicted = light_predict_trans(light_model.cuda(), src, word2id)

        for elem in src:
            question = dpros.decode(sequence=elem.numpy(), lookup=id2word, separator=' ')
            sources.append(question)

        for elem2 in predicted:
            fake = dpros.decode(sequence=elem2.cpu().numpy(), lookup=id2word, separator=' ')
            predict.append(fake)
            
    return sources, predict
            
    


## this function calculates and shows the BLEU scores for the test dataset:

def bleu_scores(answers, predict):

    smoothie = SmoothingFunction().method3

    answers_test = answers[10:15]
    predict_test = predict[10:15]
    
    for answer_test, predicted_test in zip(answers_test, predict_test):
    
        print("Answer: ", answer_test)
        print("Predicted: ", predicted_test)
    
    bleu_scores = list()
    scores_1 = list()
    scores_2 = list()
    scores_3 = list()
    scores_4 = list()

    for answer, predicted in zip(answers, predict):

        score_ref_a = bleu.sentence_bleu([answer.split()], predicted.split(), smoothing_function=smoothie)
        bleu_scores.append(score_ref_a)

        score_1gram = bleu.sentence_bleu([answer.split()], predicted.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
        scores_1.append(score_1gram)
        score_2gram = bleu.sentence_bleu([answer.split()], predicted.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        scores_2.append(score_2gram)
        score_3gram = bleu.sentence_bleu([answer.split()], predicted.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        scores_3.append(score_3gram)
        score_4gram = bleu.sentence_bleu([answer.split()], predicted.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        scores_4.append(score_4gram)

    scores_mean = statistics.mean(bleu_scores)
    print("Blue scores average: ", scores_mean)

    scores_1gram = statistics.mean(scores_1)
    print("1-gram scores average: ", scores_1gram)

    scores_2gram = statistics.mean(scores_2)
    print("2-gram scores average: ", scores_2gram)

    scores_3gram = statistics.mean(scores_3)
    print("3-gram scores average: ", scores_3gram)

    scores_4gram = statistics.mean(scores_4)
    print("4-gram scores average: ", scores_4gram)



## this function calculates and shows the ROUGE scores for the test dataset:

def rouge_scores(answers, predict):


    answers_test = answers[100:105]
    predict_test = predict[100:105]
    
    for answer_test, predicted_test in zip(answers_test, predict_test):
    
        print("Answer: ", answer_test)
        print("Predicted: ", predicted_test)

    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               apply_avg=apply_avg,
                               apply_best=apply_best,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)


        all_hypothesis = predict
        all_references = answers

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id], metric))
                print()
            else:
                print(prepare_results(results['p'], results['r'], results['f'], metric))
        print()
        
        
        

## this function calculates and shows the ROUGE scores for the test dataset:

def word_embedding_scores(answers, predict, light_model, word2id):


    answers_test = answers[200:205]
    predict_test = predict[200:205]
    
    for answer_test, predicted_test in zip(answers_test, predict_test):
        
        print("Answer: ", answer_test)
        print("Predicted: ", predicted_test)

    cos_sim = list()

    for answer, predicted in zip(answers, predict):
        
        predicted = predicted.split(' ')
        answer = answer.split(' ')
        
        predicted = zero_pad_test(predicted, word2id)
        answer = zero_pad_test(answer, word2id)
        
        predicted_embedding = light_model.embeddings(torch.tensor(predicted).cuda())
        answer_embedding = light_model.embeddings(torch.tensor(answer).cuda())
        
        predicted_mean = np.mean(predicted_embedding.cpu().detach().numpy(), axis=0)
        target_mean = np.mean(answer_embedding.cpu().detach().numpy(), axis=0)

        dot = np.dot(predicted_mean, target_mean)
        norma = np.linalg.norm(predicted_mean)
        normb = np.linalg.norm(target_mean)
        cos = dot / (norma * normb)

        cos_sim.append(float(cos))

    cos_sim_mean = statistics.mean(cos_sim)
    print("Cosine similarity average: ", cos_sim_mean)
    
    

## this function calculates and shows the word embedding similarity for the test dataset:

def word_embedding_scores_trans(answers, predict, light_model, word2id):


    answers_test = answers[200:205]
    predict_test = predict[200:205]
    
    for answer_test, predicted_test in zip(answers_test, predict_test):
        
        print("Answer: ", answer_test)
        print("Predicted: ", predicted_test)

    cos_sim = list()

    for answer, predicted in zip(answers, predict):
        
        predicted = predicted.split(' ')
        answer = answer.split(' ')
        
        predicted = zero_pad_test(predicted, word2id)
        answer = zero_pad_test(answer, word2id)
        
        predicted_embedding = light_model.embedding(torch.tensor(predicted).cuda())
        answer_embedding = light_model.embedding(torch.tensor(answer).cuda())
        
        predicted_mean = np.mean(predicted_embedding.cpu().detach().numpy(), axis=0)
        target_mean = np.mean(answer_embedding.cpu().detach().numpy(), axis=0)

        dot = np.dot(predicted_mean, target_mean)
        norma = np.linalg.norm(predicted_mean)
        normb = np.linalg.norm(target_mean)
        cos = dot / (norma * normb)

        cos_sim.append(float(cos))

    cos_sim_mean = statistics.mean(cos_sim)
    print("Cosine similarity average: ", cos_sim_mean)
  
  