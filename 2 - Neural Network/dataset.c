#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

Sample xor_probs[] = {{(float[]){0, 0}, (float[]){0, 1}},
                      {(float[]){0, 1}, (float[]){1, 0}},
                      {(float[]){1, 0}, (float[]){1, 0}},
                      {(float[]){1, 1}, (float[]){0, 1}}};

Sample xor_binary[] = {{(float[]){0, 0}, (float[]){0}},
                       {(float[]){0, 1}, (float[]){1}},
                       {(float[]){1, 0}, (float[]){1}},
                       {(float[]){1, 1}, (float[]){0}}};

// 0 means it is fraud, 1 means it is safe
Sample fraud[] = {
    {(float[]){1.169397139419055f, 13.81043565827701f}, (float[]){0}},
    {(float[]){6.938631834955604f, 15.584932585017079f}, (float[]){1}},
    {(float[]){17.303192323949517f, 0.9445826001125157f}, (float[]){1}},
    {(float[]){0.518789997982795f, 1.491224023408828f}, (float[]){1}},
    {(float[]){4.9421463664185366f, 12.096890778050575f}, (float[]){0}},
    {(float[]){1.2185581606890432f, 14.247000053464486f}, (float[]){0}},
    {(float[]){7.254525455150804f, 3.0154909634325433f}, (float[]){1}},
    {(float[]){15.9338470709928f, 12.295271490015722f}, (float[]){1}},
    {(float[]){7.394745437685092f, 3.017019760320696f}, (float[]){1}},
    {(float[]){14.006190629246369f, 9.234644384999847f}, (float[]){1}},
    {(float[]){1.669702910702413f, 19.605265576945182f}, (float[]){0}},
    {(float[]){20.81770761583412f, 16.870383156136135f}, (float[]){1}},
    {(float[]){3.1977724620419457f, 13.838295671058336f}, (float[]){0}},
    {(float[]){4.2749918774279605f, 9.255978556553616f}, (float[]){1}},
    {(float[]){11.902309195827739f, 18.335472411320268f}, (float[]){1}},
    {(float[]){20.728791469397365f, 14.651349746165522f}, (float[]){1}},
    {(float[]){18.214652052290745f, 0.04201001785680614f}, (float[]){1}},
    {(float[]){23.291643008379783f, 13.458584999533878f}, (float[]){1}},
    {(float[]){18.223261266853267f, 11.574920812359998f}, (float[]){1}},
    {(float[]){9.222000770163106f, 9.580110511929838f}, (float[]){1}},
    {(float[]){9.80923990664958f, 8.72238275017448f}, (float[]){1}},
    {(float[]){17.12065026922749f, 7.2661891081435055f}, (float[]){1}},
    {(float[]){6.49607454415521f, 5.6828654436303605f}, (float[]){1}},
    {(float[]){20.498468927792256f, 15.440510010606264f}, (float[]){1}},
    {(float[]){21.915935270683335f, 14.316374712441318f}, (float[]){1}},
    {(float[]){18.258145671950547f, 3.256618294933493f}, (float[]){1}},
    {(float[]){12.400044906193589f, 15.214155459883472f}, (float[]){1}},
    {(float[]){4.028058365316176f, 15.71656337275752f}, (float[]){0}},
    {(float[]){7.168614998612776f, 14.772131937376125f}, (float[]){1}},
    {(float[]){6.814633080735212f, 4.576030729257699f}, (float[]){1}},
    {(float[]){7.853101838825543f, 6.6042296269336624f}, (float[]){1}},
    {(float[]){11.043368011031426f, 5.522644417104869f}, (float[]){1}},
    {(float[]){13.064840626983303f, 17.951968159500183f}, (float[]){1}},
    {(float[]){5.217620874233327f, 18.50245185244278f}, (float[]){0}},
    {(float[]){19.196839846535436f, 12.371511304581093f}, (float[]){1}},
    {(float[]){17.44042678311685f, 2.7860716732793f}, (float[]){1}},
    {(float[]){20.383025121389206f, 2.750581508331602f}, (float[]){1}},
    {(float[]){6.420523309969735f, 7.317359827507202f}, (float[]){1}},
    {(float[]){14.755793067562431f, 2.309390754772054f}, (float[]){1}},
    {(float[]){15.666145756655737f, 10.236653580706177f}, (float[]){1}},
    {(float[]){9.283594419222855f, 5.187123229887137f}, (float[]){1}},
    {(float[]){9.848500285747651f, 2.209557844387222f}, (float[]){1}},
    {(float[]){19.44378118782329f, 3.76228106143053f}, (float[]){1}},
    {(float[]){14.408862369171237f, 6.479183324202036f}, (float[]){1}},
    {(float[]){21.560168126032664f, 13.310782865521261f}, (float[]){1}},
    {(float[]){11.760773401176433f, 6.239183440015683f}, (float[]){1}},
    {(float[]){19.879635173190763f, 10.478975092296498f}, (float[]){1}},
    {(float[]){20.61334730803773f, 6.67536834600611f}, (float[]){1}},
    {(float[]){18.92268344738623f, 1.7666972937182446f}, (float[]){1}},
    {(float[]){21.30373049297287f, 1.765715096242253f}, (float[]){1}},
    {(float[]){16.772658934859717f, 5.045857501704864f}, (float[]){1}},
    {(float[]){10.90536738265799f, 4.3651789026425325f}, (float[]){1}},
    {(float[]){7.341963954932532f, 1.2537837880263836f}, (float[]){1}},
    {(float[]){19.682025718222242f, 15.555541408728207f}, (float[]){1}},
    {(float[]){3.351951730176479f, 9.922642628910252f}, (float[]){1}},
    {(float[]){9.631115963536526f, 5.485012417663393f}, (float[]){1}},
    {(float[]){5.549095558568056f, 16.196666097349787f}, (float[]){1}},
    {(float[]){14.170986325040557f, 6.75352239708825f}, (float[]){1}},
    {(float[]){21.2704333904435f, 3.9122078325789267f}, (float[]){1}},
    {(float[]){7.865798169900561f, 6.839399065754712f}, (float[]){1}},
    {(float[]){19.31063164847155f, 13.455396909453494f}, (float[]){1}},
    {(float[]){19.138868147231555f, 5.662288871039718f}, (float[]){1}},
    {(float[]){15.976647546580676f, 5.313113205690416f}, (float[]){1}},
    {(float[]){5.858699921346563f, 17.62282346990982f}, (float[]){1}},
    {(float[]){19.664549898109527f, 10.78259858509967f}, (float[]){1}},
    {(float[]){11.222428570754776f, 14.875302699865983f}, (float[]){1}},
    {(float[]){17.738609447618146f, 17.95617014970672f}, (float[]){1}},
    {(float[]){11.05551523974087f, 5.47922184178182f}, (float[]){1}},
    {(float[]){20.738057989751383f, 10.668710403666278f}, (float[]){1}},
    {(float[]){16.85194300809872f, 2.663868838773744f}, (float[]){1}},
    {(float[]){4.031401762390966f, 19.237103815489448f}, (float[]){0}},
    {(float[]){4.301915355004047f, 16.249118551475682f}, (float[]){1}},
    {(float[]){20.39772068520218f, 13.028179965720039f}, (float[]){1}},
    {(float[]){8.76269392420765f, 4.045149011542184f}, (float[]){1}},
    {(float[]){9.144843345861041f, 19.416456058165167f}, (float[]){1}},
    {(float[]){0.988979766387379f, 0.7107207657266268f}, (float[]){0}},
    {(float[]){15.219913845953828f, 8.464684525717628f}, (float[]){1}},
    {(float[]){19.30028021868313f, 18.888284853628324f}, (float[]){1}},
    {(float[]){21.535509543384105f, 6.384194917098538f}, (float[]){1}},
    {(float[]){4.674323580690159f, 15.538285941575353f}, (float[]){1}},
    {(float[]){20.576216649691794f, 19.50183200019739f}, (float[]){1}},
    {(float[]){4.241509505537404f, 14.038691810716903f}, (float[]){1}},
    {(float[]){8.964506367608317f, 6.050581461318034f}, (float[]){1}},
    {(float[]){9.174690265369252f, 19.373278094009518f}, (float[]){1}},
    {(float[]){13.695480528372553f, 7.91426385957291f}, (float[]){1}},
    {(float[]){12.267281196988385f, 19.02618126671055f}, (float[]){1}},
    {(float[]){12.697855409827074f, 2.588120753775438f}, (float[]){1}},
    {(float[]){12.556864776263318f, 18.741283411054404f}, (float[]){1}},
    {(float[]){22.36506138223216f, 15.142160615362172f}, (float[]){1}},
    {(float[]){5.917822416166629f, 0.9223856320810708f}, (float[]){1}},
    {(float[]){4.999273224072193f, 13.257442896867666f}, (float[]){1}},
    {(float[]){19.584729362170727f, 4.531424582933523f}, (float[]){1}},
    {(float[]){7.292807873168513f, 10.389012711397687f}, (float[]){1}},
    {(float[]){16.2094173819089f, 10.213505930039279f}, (float[]){1}},
    {(float[]){22.16779315786074f, 2.1508385527033425f}, (float[]){1}},
    {(float[]){1.679266072710262f, 16.301268503581472f}, (float[]){0}},
    {(float[]){18.76246662245994f, 19.249060341646288f}, (float[]){1}},
    {(float[]){0.8424717364683127f, 11.77764785999159f}, (float[]){0}},
    {(float[]){12.217911903347067f, 8.085041013294504f}, (float[]){1}},
    {(float[]){10.53100016219056f, 13.117539847625217f}, (float[]){1}},
};

/**
 * @brief Creates and initializes a dataset
 *
 * @param num_samples Number of samples in this dataset
 * @param num_inputs Dimension of the samples
 * @param num_ouputs Dimension of the output of the samples
 * @return Dataset* Pointer to the created dataset, NULL if allocation fails
 */
Dataset *create_dataset(int num_samples, int num_inputs, int num_outputs) {
  Dataset *dataset = malloc(sizeof(Dataset));
  if (dataset == NULL) {
    fprintf(stderr, "Memory allocation failed for dataset creation.\n");
    return NULL;
  }

  dataset->samples = malloc(num_samples * sizeof(Sample));
  if (dataset->samples == NULL) {
    fprintf(stderr, "Memory allocation failed for samples creation.\n");
    free(dataset);
    return NULL;
  }

  for (int s = 0; s < num_samples; s++) {
    dataset->samples[s].inputs = malloc(num_inputs * sizeof(float));
    if (dataset->samples[s].inputs == NULL) {
      fprintf(stderr, "Memory allocation failed for inputs creation.\n");

      for (int i = 0; i < s; i++) {
        free(dataset->samples[i].inputs);
      }
      free(dataset->samples[s].inputs);
      free(dataset->samples);
      free(dataset);
      return NULL;
    }

    dataset->samples[s].expected_outputs = malloc(num_outputs * sizeof(float));
    if (dataset->samples[s].expected_outputs == NULL) {
      fprintf(stderr, "Memory allocation failed for outputs creation.\n");

      for (int i = 0; i < s; i++) {
        free(dataset->samples[i].expected_outputs);
        free(dataset->samples[i].inputs);
      }
      free(dataset->samples[s].inputs);
      free(dataset->samples);
      free(dataset);
      return NULL;
    }
  }

  dataset->num_samples = num_samples;

  return dataset;
}

/**
 * @brief Frees memory allocated for dataset
 *
 * @param dataset Pointer to dataset to be freed
 */
void free_dataset(Dataset *dataset) {
  for (int s = 0; s < dataset->num_samples; s++) {
    free(dataset->samples[s].inputs);
    free(dataset->samples[s].expected_outputs);
  }
  free(dataset->samples);
  free(dataset);
}

/**
 * @brief Creates an example dataset that can be used for training the model
 *
 * @return Dataset* Pointer to the dataset that has been created
 */
Dataset *create_example_dataset(char *dataset_name) {
  int NUM_SAMPLES;
  int NUM_INPUTS;
  int NUM_OUTPUTS;
  Sample *samples;
  if (strcmp(dataset_name, "xor_binary") == 0) {
    NUM_SAMPLES = 4;
    NUM_INPUTS = 2;
    NUM_OUTPUTS = 1;
    samples = xor_binary;
  } else if (strcmp(dataset_name, "xor_probs") == 0) {
    NUM_SAMPLES = 4;
    NUM_INPUTS = 2;
    NUM_OUTPUTS = 2;
    samples = xor_probs;
  } else if (strcmp(dataset_name, "fraud") == 0) {
    NUM_SAMPLES = 100;
    NUM_INPUTS = 2;
    NUM_OUTPUTS = 1;
    samples = fraud;
  }

  Dataset *dataset = create_dataset(NUM_SAMPLES, NUM_INPUTS, NUM_OUTPUTS);
  if (dataset == NULL) {
    return NULL;
  }

  for (int i = 0; i < NUM_SAMPLES; i++) {
    dataset->samples[i] = samples[i];
  }

  return dataset;
}
