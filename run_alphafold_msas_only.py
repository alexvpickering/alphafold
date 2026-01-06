# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaFold MSA generation script (features only)."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Union

from absl import app
from absl import flags
from absl import logging
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths',
    None,
    'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.',
)

flags.DEFINE_string(
    'data_dir',
    None,
    'Path to directory containing all databases. Individual database paths '
    'can override these defaults.',
)
flags.DEFINE_string(
    'output_dir', None, 'Path to a directory that will store the results.'
)
flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the JackHMMER executable.',
)
flags.DEFINE_string(
    'hhblits_binary_path',
    shutil.which('hhblits'),
    'Path to the HHblits executable.',
)
flags.DEFINE_string(
    'hhsearch_binary_path',
    shutil.which('hhsearch'),
    'Path to the HHsearch executable.',
)
flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the hmmsearch executable.',
)
flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the hmmbuild executable.',
)
flags.DEFINE_string(
    'kalign_binary_path',
    shutil.which('kalign'),
    'Path to the Kalign executable.',
)
flags.DEFINE_string(
    'uniref90_database_path',
    None,
    'Path to the Uniref90 database for use by JackHMMER. '
    'If not set, uses <data_dir>/uniref90/uniref90.fasta',
)
flags.DEFINE_string(
    'mgnify_database_path',
    None,
    'Path to the MGnify database for use by JackHMMER. '
    'If not set, uses <data_dir>/mgnify/mgy_clusters_2022_05.fa',
)
flags.DEFINE_string(
    'bfd_database_path',
    None,
    'Path to the BFD database for use by HHblits. '
    'If not set, uses <data_dir>/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
)
flags.DEFINE_string(
    'small_bfd_database_path',
    None,
    'Path to the small version of BFD used with the "reduced_dbs" preset. '
    'If not set, uses <data_dir>/small_bfd/bfd-first_non_consensus_sequences.fasta',
)
flags.DEFINE_string(
    'uniref30_database_path',
    None,
    'Path to the UniRef30 database for use by HHblits. '
    'If not set, uses <data_dir>/uniref30/UniRef30_2021_03',
)
flags.DEFINE_string(
    'uniprot_database_path',
    None,
    'Path to the Uniprot database for use by JackHMMer. '
    'If not set, uses <data_dir>/uniprot/uniprot.fasta',
)
flags.DEFINE_string(
    'pdb70_database_path',
    None,
    'Path to the PDB70 database for use by HHsearch. '
    'If not set, uses <data_dir>/pdb70/pdb70',
)
flags.DEFINE_string(
    'pdb_seqres_database_path',
    None,
    'Full filepath to the PDB seqres database file for use by hmmsearch. '
    'If not set, uses <data_dir>/pdb_seqres/pdb_seqres.txt',
)
flags.DEFINE_string(
    'template_mmcif_dir',
    None,
    'Path to a directory with template mmCIF structures, each named <pdb_id>.cif. '
    'If not set, uses <data_dir>/pdb_mmcif/mmcif_files',
)
flags.DEFINE_string(
    'max_template_date',
    None,
    'Maximum template release date '
    'to consider. Important if folding historical test sets.',
)
flags.DEFINE_string(
    'obsolete_pdbs_path',
    None,
    'Path to file containing a mapping from obsolete PDB IDs to the PDB IDs '
    'of their replacements. If not set, uses <data_dir>/pdb_mmcif/obsolete.dat',
)
flags.DEFINE_enum(
    'db_preset',
    'full_dbs',
    ['full_dbs', 'reduced_dbs'],
    'Choose preset MSA database configuration - '
    'smaller genetic database config (reduced_dbs) or '
    'full genetic database config  (full_dbs)',
)
flags.DEFINE_enum(
    'model_preset',
    'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Choose preset model configuration - the monomer model, '
    'the monomer model with extra ensembling, monomer model with '
    'pTM head, or multimer model',
)
flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed for the data '
    'pipeline. By default, this is randomly generated. Note '
    'that even if this is set, Alphafold may still not be '
    'deterministic, because processes like GPU inference are '
    'nondeterministic.',
)
flags.DEFINE_boolean(
    'use_precomputed_msas',
    False,
    'Whether to read MSAs that '
    'have been written to disk instead of running the MSA '
    'tools. The MSA files are looked up in the output '
    'directory, so it must stay the same between multiple '
    'runs that are to reuse the MSAs. WARNING: This will not '
    'check if the sequence, database or configuration have '
    'changed.',
)
flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Jackhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
flags.DEFINE_integer(
    'hmmsearch_n_cpu',
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for HMMsearch. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
flags.DEFINE_integer(
    'hhsearch_n_cpu',
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for HHsearch. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20


def _get_database_path(flag_value, data_dir, relative_path):
  """Returns flag value if set, otherwise constructs path from data_dir."""
  if flag_value:
    return flag_value
  if data_dir:
    return os.path.join(data_dir, relative_path)
  return None


def generate_features(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
):
  """Generates MSAs and features using AlphaFold data pipeline."""
  logging.info('Generating features for %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
  )
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  logging.info('Feature generation timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))

  logging.info('Features saved to %s', features_output_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer',
      'hhblits',
      'hhsearch',
      'hmmsearch',
      'hmmbuild',
      'kalign',
  ):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(
          f'Could not find path to the "{tool_name}" binary. Make '
          'sure it is installed on your system.'
      )

  # Set database paths from data_dir if not explicitly provided
  data_dir = FLAGS.data_dir
  
  uniref90_database_path = _get_database_path(
      FLAGS.uniref90_database_path, data_dir, 'uniref90/uniref90.fasta'
  )
  mgnify_database_path = _get_database_path(
      FLAGS.mgnify_database_path, data_dir, 'mgnify/mgy_clusters_2022_05.fa'
  )
  bfd_database_path = _get_database_path(
      FLAGS.bfd_database_path, data_dir, 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
  )
  small_bfd_database_path = _get_database_path(
      FLAGS.small_bfd_database_path, data_dir, 'small_bfd/bfd-first_non_consensus_sequences.fasta'
  )
  uniref30_database_path = _get_database_path(
      FLAGS.uniref30_database_path, data_dir, 'uniref30/UniRef30_2021_03'
  )
  uniprot_database_path = _get_database_path(
      FLAGS.uniprot_database_path, data_dir, 'uniprot/uniprot.fasta'
  )
  pdb70_database_path = _get_database_path(
      FLAGS.pdb70_database_path, data_dir, 'pdb70/pdb70'
  )
  pdb_seqres_database_path = _get_database_path(
      FLAGS.pdb_seqres_database_path, data_dir, 'pdb_seqres/pdb_seqres.txt'
  )
  template_mmcif_dir = _get_database_path(
      FLAGS.template_mmcif_dir, data_dir, 'pdb_mmcif/mmcif_files'
  )
  obsolete_pdbs_path = _get_database_path(
      FLAGS.obsolete_pdbs_path, data_dir, 'pdb_mmcif/obsolete.dat'
  )

  # Verify required paths are set
  if not uniref90_database_path:
    raise ValueError('uniref90_database_path must be set via flag or data_dir')
  if not mgnify_database_path:
    raise ValueError('mgnify_database_path must be set via flag or data_dir')
  if not template_mmcif_dir:
    raise ValueError('template_mmcif_dir must be set via flag or data_dir')
  if not obsolete_pdbs_path:
    raise ValueError('obsolete_pdbs_path must be set via flag or data_dir')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  if use_small_bfd and not small_bfd_database_path:
    raise ValueError('small_bfd_database_path required for reduced_dbs preset')
  if not use_small_bfd:
    if not bfd_database_path:
      raise ValueError('bfd_database_path required for full_dbs preset')
    if not uniref30_database_path:
      raise ValueError('uniref30_database_path required for full_dbs preset')

  run_multimer_system = 'multimer' in FLAGS.model_preset
  if not run_multimer_system and not pdb70_database_path:
    raise ValueError('pdb70_database_path required for monomer models')
  if run_multimer_system:
    if not pdb_seqres_database_path:
      raise ValueError('pdb_seqres_database_path required for multimer models')
    if not uniprot_database_path:
      raise ValueError('uniprot_database_path required for multimer models')

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=pdb_seqres_database_path,
        cpu=FLAGS.hmmsearch_n_cpu,
    )
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path,
    )
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[pdb70_database_path],
        cpu=FLAGS.hhsearch_n_cpu,
    )
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path,
    )

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=uniref90_database_path,
      mgnify_database_path=mgnify_database_path,
      bfd_database_path=bfd_database_path,
      uniref30_database_path=uniref30_database_path,
      small_bfd_database_path=small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      msa_tools_n_cpu=FLAGS.jackhmmer_n_cpu,
  )

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        jackhmmer_n_cpu=FLAGS.jackhmmer_n_cpu,
    )
  else:
    data_pipeline = monomer_data_pipeline

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Generate features for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    generate_features(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
    )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'max_template_date',
  ])

  app.run(main)
